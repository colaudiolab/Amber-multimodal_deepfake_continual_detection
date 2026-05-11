# -*- coding: utf-8 -*-
import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
import torch.nn as nn
from torch.nn import DataParallel
from .Learner import Learner, loader_t
import numpy as np
from torch.utils.data import DataLoader
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
        outputs = {'logits':x, 'video':video_out, 'audio':audio_out, 'features':out['features']}
        return outputs
    def feature(self, video, audio):
        out = self.backbone(video, audio)
        return out['features']

    def update_fc(self, nb_classes, device):
        fc = torch.nn.Linear(self.backbone_output, nb_classes)
        if self.fc is not None:
            nb_output = self.nb_classes
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc.to(device, non_blocking=True)
        self.nb_classes = nb_classes

class DyCRLearner(Learner):
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
        self.exemplar = None
        self.memory_per_domain: int = args["memory_per_domain"]
        self.num_features = 256
        self.dim = 4
        self.CL_type: str = CL_type
        self.nb_classes: int = 0
        self.triplet_loss = nn.TripletMarginLoss(margin=1.2).to(self.device)

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
                    # X: torch.Tensor = X.to(self.device, non_blocking=True)
                    video_label, audio_label, y = y
                    y: torch.Tensor = y.to(self.device, non_blocking=True)
                    video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                    audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)
                    assert y.max() < baseset_size

                    optimizer.zero_grad(set_to_none=True)
                    outs = model(video, audio)
                    video_out, audio_out, logits, feature = outs['video'], outs['audio'], outs['logits'], outs['features']
                    
                    class_prototypes = model.fc.weight.data
                    proto_list, answer = self._decompose(feature, y, class_prototypes)
                    tri_loss = self._cal_trip_loss(feature, proto_list, answer, y, class_prototypes)
                    
                    loss1 = criterion(logits, y)
                    loss = loss1 + tri_loss
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            # Validation on training set
            model.eval()
            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                self.save_object(
                    # (self.backbone, self.backbone_output),
                    # "backbone.pth",
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
        self.model = self.load_object(model, "model.pth")

        self.model, self.exemplar = self._replace_base_fc(train_loader, self.model, save_samples=self.memory_per_domain, dim=self.dim)

    
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
        self.update_model(phase=phase, nb_classes=nb_classes, device=self.device)

        params = self.model.parameters()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        IL_batch_size = dataset.IL_batch_size
        num_workers = dataset.num_workers
        dataset = dataset.subset_at_phase(phase, self.memory)
        data_loader = DataLoader(
            dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=False
        )
        self.exemplar = self.update_fc(data_loader, self.exemplar, dim=self.dim)
        self.model.train()
        for epoch in range(self.base_epochs):
            for indecs, X, y, _ in tqdm(data_loader, desc=desc, leave=False, ncols=50):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                # X: torch.Tensor = X.to(self.device, non_blocking=True)
                video_label, audio_label, y = y
                y: torch.Tensor = y.to(self.device, non_blocking=True)
                video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                # self.model.fit(video, audio, y, increase_size=incremental_size)
                outs = self.model(video, audio)
                video_out, audio_out, logits, feature = outs['video'], outs['audio'], outs['logits'], outs['features']
                feature = feature.detach()

                class_prototypes = self.model.fc.weight.data.detach()
                proto_list, answer = self._decompose(feature, y, class_prototypes)
                tri_loss = self._cal_trip_loss(feature, proto_list, answer, y, class_prototypes)

                # During incremental learning phases.
                if self.exemplar is not None:
                    target_labels = []

                    proto = self.model.fc.weight.detach().cpu()
                    for class_index in range(2):
                        # num = self.exemplar[class_index, :].shape[0]
                        # tmp_feature = torch.reshape(self.exemplar[class_index, :],
                        num = self.exemplar[class_index].shape[0]
                        tmp_feature = torch.reshape(self.exemplar[class_index],
                                                [num, self.dim, self.dim])
                        temp = proto[class_index]
                        skeleton = torch.reshape(temp, [self.dim, int(self.num_features/self.dim)])

                        result = torch.reshape(torch.matmul(tmp_feature, skeleton), [num, self.num_features])
                        if class_index == 0:
                            recovery = result
                        else:
                            recovery = torch.cat((recovery, result), 0)
                        target_labels += num * [class_index]
                    recovery = recovery.float().to(self.device)

                    # Combine recovered samples with current samples
                    feature = torch.cat((recovery, feature), 0)
                    logits = nn.functional.linear(nn.functional.normalize(feature, p=2, dim=-1), nn.functional.normalize(self.model.fc.weight, p=2, dim=-1))
                    temperature = 16
                    logits = temperature * logits

                    y = target_labels + y.tolist()
                    y = np.array(y)
                    y = torch.from_numpy(y).long().to(self.device)

                else:
                    logits = logits

                loss = criterion(logits, y) + tri_loss
                loss.backward()
                optimizer.step()
            scheduler.step()

        


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
  
    def load_model(self, baseset_size, state_dict, data_loader):
        print('loading pretrained model')
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model.load_state_dict(state_dict)
        model = self.wrap_data_parallel(model)
        self.model = model

        self.model, self.exemplar = self._replace_base_fc(data_loader, self.model, save_samples=self.memory_per_domain, dim=self.dim)

    def update_model(self, phase=0, nb_classes=2, device="cpu"):
        if phase > 0:
            if self.CL_type == 'CIL':
                self.nb_classes = nb_classes
                self.model.update_fc(self.nb_classes, device)
        else:
            model = FusionModel(self.backbone, self.backbone_output, nb_classes).to(self.device, non_blocking=True)
            model = self.wrap_data_parallel(model)
            self.model = model
            self.model.eval()
    
    def _decompose(self, features, label, class_prototypes):
        num_features = 256
        dim = 4
        combine = features.detach().cpu().numpy()
        combine = np.reshape(combine, (-1, dim, int(num_features/dim)))
        items = []
        answer = []

        for item in range(combine.shape[0]):
            # Decompose into category map (R) and context map (Q).
            Q, R = np.linalg.qr(combine[item], mode="complete")

            # Reshape context map back to feature shape.
            skeleton = torch.from_numpy(R.reshape(num_features)).float()
            # items.append(skeleton.view(1, num_features).requires_grad_().to(self.device))
            items.append(skeleton.view(1, num_features).to(self.device))

            # Obtain corresponding class prototype.
            answer.append(class_prototypes[label[item]].view(1, num_features))

        proto_list = torch.cat(items, dim=0)
        answer = torch.cat(answer, dim=0)

        return proto_list, answer
    
    def _cal_trip_loss(self, features, proto_list, answer, label, class_prototypes):
        # Calculate L2 distance between each sample's category information with class prototypes
        dist_map = torch.cdist(features.detach().view(features.shape[0], -1), class_prototypes.view(class_prototypes.shape[0], -1), p=2).to(self.device)
        # tri_loss = torch.tensor([0.], requires_grad=True).to(self.device)
        tri_loss = torch.tensor(0.).to(self.device)
        # tri_loss = 0.

        for i in range(features.shape[0]):
            # Obtain the first two most similar prototypes for each sample
            values, indices = torch.topk(dist_map[i], 2, largest=False)
            ground_truth = label[i].item()

            if indices[0].item() == ground_truth:
                tri_loss = tri_loss + self.triplet_loss(proto_list[i].unsqueeze(0), answer[i].unsqueeze(0)
                                                        , class_prototypes[indices[1].item()].unsqueeze(0))
            else:
                tri_loss = tri_loss + self.triplet_loss(proto_list[i].unsqueeze(0), answer[i].unsqueeze(0)
                                                        , class_prototypes[indices[0].item()].unsqueeze(0))

        tri_loss = tri_loss / (features.shape[0])
        return tri_loss
    
    def update_fc(self, data_loader, exemplar=None, dim=4):
        data = []
        label = []
        for indecs, X, y, _ in tqdm(data_loader):
            video, audio = X
            video_label, audio_label, y = y
            video = video.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            outs = self.model(video, audio)
            embedding = outs['features']
            data.append(embedding.detach().cpu())
            label.append(y.cpu())
        data = torch.cat(data, dim=0)
        label = torch.cat(label, dim=0)

        class_list = np.unique(label.numpy())
        if exemplar == None:
            new_fc, _ = self.update_fc_avg(data, label, class_list)
            return
        else:
            new_fc, exemplar = self.update_fc_avg(data, label, class_list, exemplar, dim=dim)

        return exemplar

    def update_fc_avg(self,data,label,class_list, exemplar=None, dim=4):
        num_features = 256
        new_fc=[]
        new_exemplar = None
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.model.fc.weight.data[class_index]=proto

            if exemplar is not None:
                # new_exemplar = torch.zeros([len(class_list), embedding.shape[0], dim * dim])
                new_exemplar = torch.zeros([embedding.shape[0], dim * dim])
                
                for item in range(embedding.shape[0]):
                    tmp = embedding.view(embedding.shape[0], dim,
                                                         int(num_features / dim))
                    Q, R = torch.linalg.qr(tmp[item, :, :], mode="complete")
                    # new_exemplar[class_index, item, :] = torch.reshape(Q, [dim * dim])
                    new_exemplar[item, :] = torch.reshape(Q, [dim * dim])

                if class_index >= len(exemplar):
                    # class incremental
                    # exemplar = torch.cat((exemplar, new_exemplar), dim=1)
                    exemplar.append(new_exemplar)
                    # print(len(exemplar))
                else:
                    # existing class: TIL or DIL
                    exemplar[class_index] = torch.cat((exemplar[class_index], new_exemplar), dim=0)

        new_fc=torch.stack(new_fc,dim=0)
        return new_fc, exemplar

    def _replace_base_fc(self, data_loader, model, save_samples, dim=4, exemplar=None):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
        num_features = 256

        embedding_list = []
        label_list = []
        num_classes = 2
        
        with torch.no_grad():
            for indecs, X, y, _ in tqdm(data_loader):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                # X: torch.Tensor = X.to(self.device, non_blocking=True)
                video_label, audio_label, y = y
                y: torch.Tensor = y.to(self.device, non_blocking=True)

                outs = model(video, audio)
                embedding = outs['features']

                embedding_list.append(embedding.cpu())
                label_list.append(y.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        if isinstance(save_samples,float):
            total_num = embedding_list.shape[0]
            save_samples = int(save_samples*total_num)
            save_samples = min(save_samples, 500)
        if exemplar == None:
            # exemplar = torch.zeros([num_classes, save_samples, dim * dim])
            exemplar = [torch.zeros([save_samples, dim * dim]) for _ in range(num_classes)]
        proto_list = []
        cos_similarity = nn.CosineSimilarity(dim=-1)
        for class_index in range(num_classes):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            class_mean = embedding_this.mean(0)
            proto_list.append(class_mean)

            cos_sim = cos_similarity(class_mean, embedding_this)
            save_far_elems = embedding_this[
                torch.topk(cos_sim, int(save_samples / 2), largest=False, sorted=False).indices]
            save_close_elems = embedding_this[
                torch.topk(cos_sim, int(save_samples / 2), largest=True, sorted=False).indices]
            save_far_elems = save_far_elems.view(int(save_samples / 2), dim,
                                                int(num_features / dim))
            save_close_elems = save_close_elems.view(int(save_samples / 2), dim,
                                                    int(num_features / dim))

            for item in range(int(save_samples / 2)):
                Qf, Rf = torch.linalg.qr(save_far_elems[item, :, :], mode="complete")
                Qc, Rc = torch.linalg.qr(save_close_elems[item, :, :], mode="complete")
                # exemplar[class_index, 2 * item, :] = torch.reshape(Qf, [dim * dim])
                # exemplar[class_index, 2 * item + 1, :] = torch.reshape(Qc, [dim * dim])
                exemplar[class_index][2 * item, :] = torch.reshape(Qf, [dim * dim])
                exemplar[class_index][2 * item + 1, :] = torch.reshape(Qc, [dim * dim])

        proto_list = torch.stack(proto_list, dim=0)

        model.fc.weight.data[:num_classes] = proto_list

        return model, exemplar