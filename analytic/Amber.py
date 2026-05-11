# -*- coding: utf-8 -*-
# v1 version: Save 100 samples per domain for each domain in the memory. Add Gausiian noise to the input and calculate the JS loss between the augmented and the original input. Select 50 hard samples and 50 easy samples for each domain according to the JS loss. 

import torch
import torch.nn.functional as F
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from torch.nn import DataParallel
from .Learner import Learner, loader_t
import numpy as np

from .mrfa import MRFA
from torch.utils.data import Dataset, DataLoader
import copy


class FusionModel(torch.nn.Module):
    def __init__(self, backbone, backbone_output, nb_classes):
        super().__init__()
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.nb_classes = nb_classes
        self.fc = torch.nn.Linear(backbone_output, nb_classes)
    def forward(self, video, audio, train=True, phase=0):
        out = self.backbone(video, audio)
        video_out, audio_out, x = out['video'], out['audio'], out['features']
        x = self.fc(x)
        outputs = {'logits':x, 'video':video_out, 'audio':audio_out}
        return outputs

class AmberLearner(Learner):
    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        data_manager,
        CL_type: str,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        super().__init__(args, backbone, backbone_output, data_manager, device, all_devices)
        self.learning_rate: float = args["learning_rate"]
        self.buffer_size: int = args["buffer_size"]
        self.gamma: float = args["gamma"]
        self.base_epochs: int = args["base_epochs"]
        self.warmup_epochs: int = args["warmup_epochs"]
        self.memory = []
        self.memory_per_domain: int = int(args["memory_per_domain"])
        self._known_domains = 0
        self.CL_type: str = CL_type
        

        p=0.00005
        self.perturb_p = np.array([p,p,p,p,p])
        self.disable_perturb = False
        self.num_augmem = 1
        self.perturb_all = False
        self.strategy='aug'
        # self.strategy = None

        print(f'memory per domain: {self.memory_per_domain}, strategy: {self.strategy}, p: {p}, disable_perturb: {self.disable_perturb}')
        self.MRFA = MRFA()

    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model = self.wrap_data_parallel(model)


        if self.args["separate_decay"]:
            params = set_weight_decay(model, self.args["weight_decay"])
        else:
            params = model.parameters()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        best_acc = 0.0
        logging_file_path = path.join(self.args["saving_root"], "base_training.csv")
        logging_file = open(logging_file_path, "w", buffering=1)
        print(
            "epoch",
            "best_acc@1",
            "loss",
            "acc@1",
            "acc@5",
            "f1-micro",
            "training_loss",
            "training_acc@1",
            "training_acc@5",
            "training_f1-micro",
            "training_learning-rate",
            file=logging_file,
            sep=",",
        )

        for epoch in range(self.base_epochs + 1):
            if epoch != 0:
                print(
                    f"Base Training - Epoch {epoch}/{self.base_epochs}",
                    f"(Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']})",
                )
                model.train()
                for indecs, X, y,_ in tqdm(train_loader, "Training"):
                    video, audio = X
                    video = video.to(self.device, non_blocking=True)
                    audio = audio.to(self.device, non_blocking=True)
                    video_label, audio_label, y = y
                    y: torch.Tensor = y.to(self.device, non_blocking=True)
                    video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                    audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)
                    assert y.max() < baseset_size

                    optimizer.zero_grad(set_to_none=True)
                    outs = model(video, audio)
                    video_out, audio_out, logits = outs['video'], outs['audio'], outs['logits']
                    # loss1 = criterion(video_out, video_label)
                    # loss2 = criterion(audio_out, audio_label)
                    # loss3 = criterion(logits, y)
                    # loss = loss1 + loss2 + loss3
                    # loss = loss3
                    loss: torch.Tensor = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            # Validation on training set
            model.eval()
            # train_meter = validate(
            #     model, train_loader, baseset_size, desc="Training (Validation)"
            # )
            # print(
            #     f"loss: {train_meter.loss:.4f}",
            #     f"acc@1: {train_meter.accuracy * 100:.3f}%",
            #     f"auc: {train_meter.auc * 100:.3f}%",
            #     f"f1-micro: {train_meter.f1_micro * 100:.3f}%",
            #     sep="    ",
            # )

            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                self.save_object(
                    model.state_dict(),
                    "model.pth"
                )

            # Validation on testing set
            print(
                f"loss: {val_meter.loss:.4f}",
                f"acc@1: {val_meter.accuracy * 100:.3f}%",
                f"auc: {val_meter.auc * 100:.3f}%",
                f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
                f"best_acc@1: {best_acc * 100:.3f}%",
                sep="    ",
            )
            print(
                epoch,
                best_acc,
                val_meter.loss,
                val_meter.accuracy,
                val_meter.auc,
                val_meter.f1_micro,
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
        logging_file.close()
        self.backbone.eval()
        self.model = model
        # self.model = self.load_object(model, "model.pth")

        self._known_domains += 1    # 可能增加的域不止1个
        # self._reduce_exemplar(train_loader, self.memory_per_class)
        self._construct_exemplar_unified(train_loader, self.memory_per_domain, strategy=self.strategy)

    
    def learn(
        self,
        # data_loader: loader_t,
        dataset,
        incremental_size: int,
        phase: int,
        nb_classes: int,
        desc: str = "Incremental Learning",
    ) -> None:
        if desc == 'Re-align': return
        if self.CL_type == 'CIL':
            self.nb_classes = nb_classes
            self.model.update_fc(self.nb_classes, self.device)

        params = self.model.parameters()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        # base_dataset = self.task_train_dataset
        base_num_samples = dataset.original_dataset_length
        IL_batch_size = dataset.IL_batch_size
        num_workers = dataset.num_workers
        dataset = dataset.subset_at_phase(phase, self.memory) # 训练集中加入learner保存的样本

        # aug_imgs, aug_targets = self._get_memory()
        # mem_aug_dataset = AugmentMemoryDataset(aug_imgs, aug_targets, base_dataset.trsf, index_offset=base_num_samples, use_path=base_dataset.use_path)

        # dataset_list = [base_dataset, *([mem_aug_dataset]*self.num_augmem)]
        # concat_dataset = ConcatDataset(dataset_list)
        data_loader = DataLoader(
            dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
        # self.MRFA.register_perturb_forward_prehook(self.model, "GAT")
        self.MRFA.register_perturb_forward_prehook(self.model, "ResNet")
        # self.MRFA.register_perturb_forward_prehook(self.model, "LTI")

        
        for epoch in range(self.base_epochs):
            for indices, X, y, _ in tqdm(data_loader, desc=desc):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                video_label, audio_label, y = y
                y: torch.Tensor = y.to(self.device, non_blocking=True)
                video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)

                if (((perturb_indices := indices - base_num_samples) >= 0).any() or self.perturb_all) and not self.disable_perturb:
                    # print('perturb_indices', perturb_indices)
                    perturb_mask = perturb_indices >= 0 if not self.perturb_all else indices >= 0
                    perturb_indices = perturb_indices[perturb_mask]
                    # self.MRFA.feature_augmentation(self.model, video[perturb_mask], audio[perturb_mask], y[perturb_mask], "GAT")
                    self.MRFA.feature_augmentation(self.model, video[perturb_mask], audio[perturb_mask], y[perturb_mask], "ResNet")
                    # self.MRFA.feature_augmentation(self.model, video[perturb_mask], audio[perturb_mask], y[perturb_mask], "LTI")

                    self.MRFA.perturbation_idices.extend(np.arange(len(perturb_indices)).tolist())

                    self.MRFA.perturbation_idices_inbatch.extend(perturb_mask.nonzero().flatten().tolist())
                    # self.MRFA.perturbation_layers.extend(np.random.randint(0, len(self.perturb_p), len(perturb_indices)).tolist())
                    layers = [3]*len(perturb_indices)
                    self.MRFA.perturbation_layers.extend(layers)
                    self.MRFA.perturbation_factor = (self.perturb_p[self.MRFA.perturbation_layers] * np.random.rand(len(perturb_indices))).tolist()

                self.model.train()
                optimizer.zero_grad(set_to_none=True)
                outs = self.model(video, audio)
                video_out, audio_out, logits = outs['video'], outs['audio'], outs['logits']
                # loss1 = criterion(video_out, video_label)
                # loss2 = criterion(audio_out, audio_label)
                # loss3 = criterion(logits, y)
                # loss = loss1 + loss2 + loss3
                # loss.backward()
                # loss1 = criterion(video_out, video_label)
                # loss2 = criterion(audio_out, audio_label)
                # loss3 = criterion(logits, y)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                self.MRFA._init_inbatch_properties()
            scheduler.step()

        self._known_domains += 1
        # self._reduce_exemplar(data_loader, self.memory_per_class)
        self._construct_exemplar_unified(data_loader, self.memory_per_domain, strategy=self.strategy)


    def before_validation(self, phase) -> None:
        self.save_object(
            self.model.state_dict(),
            f"model_{phase}.pth"
        )

    def inference(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.model(video, audio)

    @torch.no_grad()
    def wrap_data_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.all_devices is not None and len(self.all_devices) > 1:
            return DataParallel(model, self.all_devices, output_device=self.device) # type: ignore
        return model

    def before_training(self, dataset):
        pass

    def _reduce_exemplar(self, dataloader, m):
        print("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        # self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # # Exemplar mean
            # idx_dataset = data_manager.get_dataset(
            #     [], source="train", mode="test", appendent=(dd, dt)
            # )
            # idx_loader = DataLoader(
            #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            # )
            # vectors, _ = self._extract_vectors(idx_loader)
            # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            # mean = np.mean(vectors, axis=0)
            # mean = mean / np.linalg.norm(mean)

            # self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, dataloader, m):
        # data, targets, idx_dataset = data_manager.get_dataset(
        #     np.arange(class_idx, class_idx + 1),
        #     source="train",
        #     mode="test",
        #     ret_data=True,
        # )
        # idx_loader = DataLoader(
        #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        # )
        # vectors, _ = self._extract_vectors(idx_loader)
        # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        # class_mean = np.mean(vectors, axis=0)

        # # Select
        # selected_exemplars = []
        # exemplar_vectors = []  # [n, feature_dim]
        # for k in range(1, m + 1):
        #     S = np.sum(
        #         exemplar_vectors, axis=0
        #     )  # [feature_dim] sum of selected exemplars vectors
        #     mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
        #     i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
        #     selected_exemplars.append(
        #         np.array(data[i])
        #     )  # New object to avoid passing by inference
        #     exemplar_vectors.append(
        #         np.array(vectors[i])
        #     )  # New object to avoid passing by inference

        #     vectors = np.delete(
        #         vectors, i, axis=0
        #     )  # Remove it to avoid duplicative selection
        #     data = np.delete(
        #         data, i, axis=0
        #     )  # Remove it to avoid duplicative selection

        # selected_exemplars = np.array(selected_exemplars)
        # exemplar_targets = np.full(m, class_idx)
        # self._data_memory = (
        #     np.concatenate((self._data_memory, selected_exemplars))
        #     if len(self._data_memory) != 0
        #     else selected_exemplars
        # )
        # self._targets_memory = (
        #     np.concatenate((self._targets_memory, exemplar_targets))
        #     if len(self._targets_memory) != 0
        #     else exemplar_targets
        # )

        # # Exemplar mean
        # idx_dataset = data_manager.get_dataset(
        #     [],
        #     source="train",
        #     mode="test",
        #     appendent=(selected_exemplars, exemplar_targets),
        # )
        # idx_loader = DataLoader(
        #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        # )
        # vectors, _ = self._extract_vectors(idx_loader)
        # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        # mean = np.mean(vectors, axis=0)
        # mean = mean / np.linalg.norm(mean)

        # self._class_means[class_idx, :] = mean

        # 随机选择m个样本作为exemplar
        exemplars = []
        for i, data in enumerate(dataloader):
            data, label, index = data
            fn_img, fn_aud, label, start = index
            exemplar = list(zip(fn_img, fn_aud, label, start))
            for i in exemplar:
                fn_img, fn_aud, label, start = i
                label = label.item()
                start = start.item()
                idx = (fn_img, fn_aud, label, start)
                exemplars.append(idx)
                if len(exemplars) >= m:
                    exemplars = exemplars[:m]
                    break
        self.memory.extend(exemplars)
        print(f'Construct exemplars: {len(exemplars)}')
            
    def _construct_exemplar_unified(self, dataloader, m, strategy=None):
        #-------------------------------------------------------------------------
        # calculate the prototype of two class
        features = []
        labels = []
        self.backbone.eval()
        with torch.no_grad():
            for i, (indecs, X, y,_) in enumerate(dataloader):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                video_label, audio_label, target = y
                feature = self.backbone(video, audio)['features']
                # feature = self.backbone(video, audio)['video']    # GPU 0
                # feature = self.backbone(video, audio)['audio']  # GPU 1
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototypes = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            feature_classwise = features[index]
            prototype = np.mean(feature_classwise, axis=0)
            # TODO: Need normalization?
            # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            # class_mean = np.mean(vectors, axis=0)
            prototypes.append(prototype)
        prototypes = torch.tensor(prototypes, device=self.device)    
        #-------------------------------------------------------------------------
        real_exemplars, fake_exemplars = [], []
        real_distances, fake_distances = [], []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                _, data, label, index = data
                video, audio = data
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                video_label, audio_label, target = label
                feature = self.backbone(video, audio)['features']
                if strategy == "aug":
                    aug_video, aug_audio = self._augment(video, audio, method="noise")
                    aug_feature = self.backbone(aug_video, aug_audio)['features']
                # real and fake sample index
                real_index = np.where(target == 0)[0]
                fake_index = np.where(target == 1)[0]

                fn_img, fn_aud, label, start = index
                exemplars = np.array(list(zip(fn_img, fn_aud, label.tolist(), start.tolist())), dtype=object)
                real_exemplars.append(exemplars[real_index])
                fake_exemplars.append(exemplars[fake_index])

                # calculate the distance of the feature
                if strategy == None:
                    real_distance = torch.sqrt(torch.sum((feature[real_index] - prototypes[0]) ** 2, axis=1))
                    fake_distance = torch.sqrt(torch.sum((feature[fake_index] - prototypes[1]) ** 2, axis=1))
                elif strategy == "aug":
                    real_distance = self.js_divergence(feature[real_index], aug_feature[real_index])
                    fake_distance = self.js_divergence(feature[fake_index], aug_feature[fake_index])
                real_distances.append(real_distance.cpu().data)
                fake_distances.append(fake_distance.cpu().data)

            # sort the distance
            real_distances = np.concatenate(real_distances)
            fake_distances = np.concatenate(fake_distances)
            if strategy == 'aug':
                real_distances_sorted = np.concatenate((np.argsort(real_distances)[-m//4:], np.argsort(real_distances)[:m//4]))
                fake_distances_sorted = np.concatenate((np.argsort(fake_distances)[-m//4:], np.argsort(fake_distances)[:m//4]))
                # real_distances_sorted = np.argsort(real_distances)[:m//2]
                # fake_distances_sorted = np.argsort(fake_distances)[:m//2]
            else:
                real_distances_sorted = np.argsort(real_distances)[:m//2]
                fake_distances_sorted = np.argsort(fake_distances)[:m//2]
            # print(np.argsort(real_distances))
            # print(real_distances_sorted)
            
            # Selection
            real_exemplars = np.concatenate(real_exemplars)
            fake_exemplars = np.concatenate(fake_exemplars)
            real_exemplars = real_exemplars[real_distances_sorted]
            fake_exemplars = fake_exemplars[fake_distances_sorted]
            selected_exemplars = np.concatenate((real_exemplars, fake_exemplars), axis=0)

            # add to memory
            self.memory = (
                np.concatenate((self.memory, selected_exemplars))
                if len(self.memory) != 0
                else selected_exemplars
            )

    def _augment(self, video, audio, method=None):
        if method == "noise":
            video_noise = torch.randn_like(video) * 0.1
            audio_noise = torch.randn_like(audio) * 0.1
            video = video + video_noise
            audio = audio + audio_noise
        return video, audio
    
    def KL_DIV(self, P, Q):
        M = torch.mean(P,Q, )
        P_log = torch.log(P)
        # 使用kl_div函数计算KL散度
        kl_div = F.kl_div(P_log, Q, reduction='none')
        kl_div = kl_div.sum(dim=1)
        return kl_div
    
    def js_divergence(self, p, q, base=2):
        """
        计算 JS 散度 (使用 PyTorch，支持 batch 计算)
        
        参数:
        - p: 第一个概率分布 (torch tensor, shape: [batch_size, num_classes])
        - q: 第二个概率分布 (torch tensor, shape: [batch_size, num_classes])
        - base: 对数的底数，默认为 2
        
        返回:
        - JS 散度值 (shape: [batch_size])
        """
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        m = 0.5 * (p + q)
        
        kl_pm = torch.sum(p * (torch.log(p / m) / torch.log(torch.tensor(base, dtype=p.dtype, device=p.device))), dim=-1)
        kl_qm = torch.sum(q * (torch.log(q / m) / torch.log(torch.tensor(base, dtype=p.dtype, device=p.device))), dim=-1)
        
        return 0.5 * (kl_pm + kl_qm)