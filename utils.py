# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import random
from glob import glob
import shutil
import requests
import tarfile


import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

### GSC
label_dict = {
    "_silence_": 0,
    "_unknown_": 1,
    "down": 2,
    "go": 3,
    "left": 4,
    "no": 5,
    "off": 6,
    "on": 7,
    "right": 8,
    "stop": 9,
    "up": 10,
    "yes": 11,
}
print("labels:\t", label_dict)
sample_per_cls_v1 = [1854, 258, 257]
sample_per_cls_v2 = [3077, 371, 408]
SR = 16000


def ScanAudioFiles(root_dir, ver):
    sample_per_cls = sample_per_cls_v1 if ver == 1 else sample_per_cls_v2
    audio_paths, labels = [], []
    for path, _, files in sorted(os.walk(root_dir, followlinks=True)):
        random.shuffle(files)
        for idx, filename in enumerate(files):
            if not filename.endswith(".wav"):
                continue
            dataset, class_name = path.split("/")[-2:]
            if class_name in ("_unknown_", "_silence_"):  # balancing
                if "train" in dataset and idx == sample_per_cls[0]:
                    break
                if "valid" in dataset and idx == sample_per_cls[1]:
                    break
                if "test" in dataset and idx == sample_per_cls[2]:
                    break
            audio_paths.append(os.path.join(path, filename))
            labels.append(label_dict[class_name])
    return audio_paths, labels


class SpeechCommand(Dataset):
    """GSC"""

    def __init__(self, root_dir, ver, transform=None):
        self.transform = transform
        self.data_list, self.labels = ScanAudioFiles(root_dir, ver)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.data_list[idx]
        sample, _ = torchaudio.load(audio_path)
        if self.transform:
            sample = self.transform(sample)
        label = self.labels[idx]
        return sample, label


def spec_augment(
    x, frequency_masking_para=20, time_masking_para=20, frequency_mask_num=2, time_mask_num=2
):
    lenF, lenT = x.shape[1:3]
    # Frequency masking
    for _ in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, lenF - f)
        x[:, f0 : f0 + f, :] = 0
    # Time masking
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, lenT - t)
        x[:, :, t0 : t0 + t] = 0
    return x


class Preprocess:
    def __init__(
        self,
        noise_loc,
        device,
        hop_length=160,
        win_length=480,
        n_fft=512,
        n_mels=40,
        specaug=False,
        sample_rate=SR,
        frequency_masking_para=7,
        time_masking_para=20,
        frequency_mask_num=2,
        time_mask_num=2,
    ):
        if noise_loc is None:
            self.background_noise = []
        else:
            self.background_noise = [
                torchaudio.load(file_name)[0] for file_name in glob(noise_loc + "/*.wav")
            ]
            assert len(self.background_noise) != 0
        self.feature = LogMel(
            device,
            sample_rate=sample_rate,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        self.sample_len = sample_rate
        self.specaug = specaug
        self.device = device
        if self.specaug:
            self.frequency_masking_para = frequency_masking_para
            self.time_masking_para = time_masking_para
            self.frequency_mask_num = frequency_mask_num
            self.time_mask_num = time_mask_num
            print(
                "frequency specaug %d %d" % (self.frequency_mask_num, self.frequency_masking_para)
            )
            print("time specaug %d %d" % (self.time_mask_num, self.time_masking_para))

    def __call__(self, x, labels, augment=True, noise_prob=0.8, is_train=True):
        assert len(x.shape) == 3
        if augment:
            for idx in range(x.shape[0]):
                if labels[idx] != 0 and (not is_train or random.random() > noise_prob):
                    continue
                noise_amp = (
                    np.random.uniform(0, 0.1) if labels[idx] != 0 else np.random.uniform(0, 1)
                )
                noise = random.choice(self.background_noise).to(self.device)
                sample_loc = random.randint(0, noise.shape[-1] - self.sample_len)
                noise = noise_amp * noise[:, sample_loc : sample_loc + SR]

                if is_train:
                    x_shift = int(np.random.uniform(-0.1, 0.1) * SR)
                    zero_padding = torch.zeros(1, np.abs(x_shift)).to(self.device)
                    if x_shift < 0:
                        temp_x = torch.cat([zero_padding, x[idx, :, :x_shift]], dim=-1)
                    else:
                        temp_x = torch.cat([x[idx, :, x_shift:], zero_padding], dim=-1)
                    x[idx] = temp_x + noise
                else:  # valid
                    x[idx] = x[idx] + noise
                x[idx] = torch.clamp(x[idx], -1.0, 1.0)

        x = self.feature(x)
        if self.specaug:
            for i in range(x.shape[0]):
                x[i] = spec_augment(
                    x[i],
                    self.frequency_masking_para,
                    self.time_masking_para,
                    self.frequency_mask_num,
                    self.time_mask_num,
                )
        return x


class LogMel:
    def __init__(
        self, device, sample_rate=SR, hop_length=160, win_length=480, n_fft=512, n_mels=40
    ):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.device = device

    def __call__(self, x):
        self.mel = self.mel.to(self.device)
        output = (self.mel(x) + 1e-6).log()
        return output


class Padding:
    """zero pad to have 1 sec len"""

    def __init__(self):
        self.output_len = SR

    def __call__(self, x):
        pad_len = self.output_len - x.shape[-1]
        if pad_len > 0:
            x = torch.cat([x, torch.zeros([x.shape[0], pad_len])], dim=-1)
        elif pad_len < 0:
            raise ValueError("no sample exceed 1sec in GSC.")
        return x

def DownloadDataset(loc, url):
    if not os.path.isdir(loc):
        os.mkdir(loc)
    filename = os.path.basename(url)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1048576
    with open(os.path.join(loc, filename), "wb") as f:
        for data in response.iter_content(block_size):
            f.write(data)
            read_so_far = f.tell()
            if total_size > 0:
                percent = read_so_far * 100 / total_size
                print(f"Downloaded {read_so_far} of {total_size} bytes ({percent:.2f}%)")
    with tarfile.open(os.path.join(loc, filename), "r:gz") as tar:
        tar.extractall(loc)

def make_empty_audio(loc, num):
    if not os.path.isdir(loc):
        os.mkdir(loc)
    for i in range(num):
        path = os.path.join(loc, "%s.wav" % str(i))
        zeros = torch.zeros([1, SR])  # 1 sec long.
        torchaudio.save(path, zeros, SR)


def make_12class_dataset(base, target):
    os.mkdir(target)
    os.mkdir(target + "/_unknown_")
    class10 = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
    for clsdir in glob(os.path.join(base, "*")):
        class_name = os.path.basename(clsdir)
        if class_name in class10:
            target_dir = os.path.join(target, class_name)
            shutil.copytree(clsdir, target_dir)
            print(f"Copied {clsdir} to {target_dir}")
        else:
            for file_path in glob(os.path.join(clsdir, "*")):
                filename = os.path.basename(file_path)
                target_dir = os.path.join(target, "_unknown_")
                os.makedirs(target_dir, exist_ok=True)
                target_file = os.path.join(target_dir, class_name + "_" + filename)
                shutil.copy(file_path, target_file)
                print(f"Copied {file_path} to {target_file}")

def split_data(base, target, valid_list, test_list):
    with open(valid_list, "r") as f:
        valid_names = [item.rstrip() for item in f.readlines()]
    with open(test_list, "r") as f:
        test_names = [item.rstrip() for item in f.readlines()]

    trg_base_dirs = [
        os.path.join(target, "train"),
        os.path.join(target, "valid"),
        os.path.join(target, "test"),
    ]
    for item in trg_base_dirs:
        if not os.path.isdir(item):
            os.mkdir(item)

    for root, _, files in os.walk(base):
        for file_name in files:
            if not file_name.endswith(".wav"):
                continue

            if "_background_noise_" in os.path.join(root, file_name):
                continue

            class_name = root.split("/")[-1]
            for item in trg_base_dirs:
                if not os.path.isdir(os.path.join(item, class_name)):
                    os.mkdir(os.path.join(item, class_name))
            org_file_name = os.path.join(root, file_name)
            trg_file_name = os.path.join(class_name, file_name)
            if trg_file_name in valid_names:
                target_dir = trg_base_dirs[1]
            elif trg_file_name in test_names:
                target_dir = trg_base_dirs[-1]
            else:
                target_dir = trg_base_dirs[0]
            target_path = os.path.join(target_dir, trg_file_name)
            shutil.copy(org_file_name, target_path)
            print(f"Copied {org_file_name} to {target_path}")


def SplitDataset(loc):
    target_loc = "%s_split" % loc
    if not os.path.isdir(target_loc):
        os.mkdir(target_loc)
    split_data(
        loc,
        target_loc,
        os.path.join(loc, "validation_list.txt"),
        os.path.join(loc, "testing_list.txt"),
    )

    sample_per_cls = sample_per_cls_v1 if "v0.01" in loc else sample_per_cls_v2
    for idx, split_name in enumerate(["train", "valid", "test"]):
        make_12class_dataset(
            "%s/%s" % (target_loc, split_name), "%s/%s_12class" % (loc, split_name)
        )
        make_empty_audio("%s/%s_12class/_silence_" % (loc, split_name), sample_per_cls[idx])
