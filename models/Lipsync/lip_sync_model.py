from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn

from .artifact_detector import ArtifactDetector
from .audio_encoder import AudioEncoder
from .classifier import ClassificationHead
from .fusion_module import CrossModalAttention, FeatureProjection
from .temporal import TemporalTransformer
from .visual_encoder import VisualEncoder
import torchaudio

class FbankExtractor(torch.nn.Module):
    def __init__(self, sr=16000, n_mels=128, frame_len=25, frame_shift=10):
        super().__init__()
        # 将毫秒转换为秒
        self.win_length = int(frame_len / 1000 * sr)
        self.hop_length = int(frame_shift / 1000 * sr)
        
        # 组合 MelSpectrogram 模块
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=512,  # 常用的 FFT 点数
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=n_mels,
            power=2
        )
    
    def forward(self, wav):
        # wav 输入形状应为 [batch, samples]
        mel = self.mel_spec(wav)          # 输出形状 [batch, n_mels, T]
        log_mel = torch.log(mel + 1e-6)   # 添加微小值防止 log(0)
        return log_mel.transpose(1, 2).unsqueeze(1)    # 转换为 [batch, T, n_mels]

# def process_audio(wav, sr = 16000):

#     wavform = wav * 2 ** 15
#     print(wavform.shape)
#     # shape[T,128]
#     fbank = ta_kaldi.fbank(wavform, num_mel_bins=128, sample_frequency=sr, frame_length=25, frame_shift=10)
#     print(fbank.shape)
#     return fbank

class LipSyncModel(nn.Module):
    """
    End‑to‑end audio‑visual lip‑sync detection model with AI manipulation detection.

    Architecture:
    - VisualEncoder + AudioEncoder (unchanged)
    - FeatureProjection → CrossModalAttention (replaces concat fusion)
    - TemporalTransformer with CLS token (replaces global avg pool)
    - ArtifactDetector branch (CLS + visual feature map → artifact features)
    - Final concat: CLS (256) + artifact (128) → ClassificationHead
    """

    def __init__(
        self,
        visual_feature_dim: int = 256,
        audio_feature_dim: int = 256,
        embed_dim: int = 256,
        detect_artifacts: bool = True,
        cross_modal_heads: int = 8,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        temporal_pre_conv: bool = True,
        use_delta_artifact: bool = True,
        use_high_freq_artifact: bool = True,
        preserve_audio_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.detect_artifacts = detect_artifacts

        # Encoders
        self.visual_encoder = VisualEncoder(feature_dim=visual_feature_dim)
        self.audio_encoder = AudioEncoder(
            feature_dim=audio_feature_dim,
            preserve_audio_temporal=preserve_audio_temporal,
        )
        self.fbank = FbankExtractor()

        # Feature projection + cross-modal attention
        self.projection = FeatureProjection(
            visual_dim=visual_feature_dim,
            audio_dim=audio_feature_dim,
            embed_dim=embed_dim,
        )
        self.cross_modal = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=cross_modal_heads,
        )

        # Temporal transformer (replaces global avg pool)
        self.temporal = TemporalTransformer(
            embed_dim=embed_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            pre_conv=temporal_pre_conv,
        )
        self.v_temporal = TemporalTransformer(
            embed_dim=embed_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            pre_conv=temporal_pre_conv,
        )
        self.a_temporal = TemporalTransformer(
            embed_dim=embed_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            pre_conv=temporal_pre_conv,
        )

        # Artifact detection
        if detect_artifacts:
            self.artifact_detector = ArtifactDetector(
                visual_feature_dim=visual_feature_dim,
                embed_dim=embed_dim,
                use_delta_map=use_delta_artifact,
                use_high_freq=use_high_freq_artifact,
            )
            classifier_input_dim = embed_dim + embed_dim // 2  # 256 + 128
        else:
            self.artifact_detector = None
            classifier_input_dim = embed_dim

        self.classifier = ClassificationHead(
            input_dim=classifier_input_dim, hidden_dim=128
        )
        self.fmap_dim = 256
        self.out_dim = classifier_input_dim
        self.v_classifier = nn.Linear(256, 384)
        self.a_classifier = nn.Linear(256, 384)

    def forward(
        self,
        visual: Tensor,
        audio: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Args:
            visual: Tensor `(B, 3, T_v, H, W)` – mouth‑crop video clip.
            audio:  Tensor `(B, 1, F, T_a)` – log Mel‑spectrogram.

        Returns:
            Tensor `(B,)` – **logits** for P(REAL). Apply `torch.sigmoid` to get probability.
        """
        B, TC, H, W = visual.shape
        visual = visual.view(B, 3, -1, H, W)
        audio = self.fbank(audio)
        # Encode modalities
        if self.detect_artifacts and self.artifact_detector is not None:
            v_feat, v_map = self.visual_encoder(visual, return_map=True)
        else:
            v_feat = self.visual_encoder(visual)
            v_map = None
        a_feat = self.audio_encoder(audio)

        # Project to shared embedding
        v_emb, a_emb = self.projection(v_feat, a_feat)

        # Cross-modal attention (replaces concat fusion)
        fused = self.cross_modal(v_emb, a_emb)  # (B, T, D_e)

        # Temporal transformer → CLS output
        cls_output = self.temporal(fused)  # (B, D_e)
        video_output = self.v_temporal(v_emb)
        audio_output = self.a_temporal(a_emb)

        # Artifact branch + final concat
        if self.detect_artifacts and self.artifact_detector is not None:
            if v_map is None:
                raise RuntimeError("Artifact detection enabled but visual feature map is missing.")
            artifact_feat = self.artifact_detector(v_map, cls_output, raw_video=visual)  # (B, 128)
            combined = torch.cat([cls_output, artifact_feat], dim=-1)  # (B, 384)
        else:
            combined = cls_output

        # logits = self.classifier(combined)
        video_output = self.v_classifier(video_output)
        audio_output = self.a_classifier(audio_output)
        out = {'video': video_output, 'audio': audio_output, 'features': combined, 'fmaps': fused}
        if not return_aux:
            return out

        aux: Dict[str, Tensor] = {
            "visual_tokens": v_emb,
            "audio_tokens": a_emb,
            "fused_tokens": fused,
            "cls_output": cls_output,
        }
        return logits, aux

    @torch.no_grad()
    def predict(self, visual: Tensor, audio: Tensor) -> Tensor:
        """
        Convenience wrapper around `forward` that ensures eval mode and
        disables gradient tracking.
        """
        self.eval()
        return self.forward(visual, audio)

if __name__ == "__main__":
    net = LipSyncModel()
    logits, aux = net(torch.randn(2, 120, 128, 128), torch.randn(2, 64000))
    print(logits)
    # print(x_video.shape, x_audio.shape, x.shape)
    # print(summary(net, (10, 512)))