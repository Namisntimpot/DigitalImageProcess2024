import os
import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader import *
from networks import *
from evaluator import *

def train_loop(dataset:FaceDataset, model:FaceNet, optimizer:torch.optim.Optimizer, n_epoches, logdir = "runs/resnet18", save_path = "facenet_resnet18.pt"):
    assert model.training_mode

    acc = 0
    los = 0
    t = 0
    writer = SummaryWriter(logdir)
    global_step = 0

    dataloader = DataLoader(dataset, 8, shuffle=True)
    for ep in range(n_epoches):
        for data, label in dataloader:
            feat, cla = model.forward(data)
            loss = model.NllLoss(cla, label)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算准确率
            _, c = torch.max(cla, 1)
            correct = (c == label).sum().item()
            accuracy = correct / label.size(0)

            writer.add_scalar("Accuracy", accuracy, global_step)
            writer.add_scalar("Loss", loss.item(), global_step)
            global_step += 1

            acc += accuracy
            los += loss.item()
            t += 1

        print("[Epoch {}] accuracy: {}, loss: {}.".format(ep, acc / t, los / t))
        acc = 0
        los = 0
        t = 0
        torch.save(model.get_feat_extractor_state_dict(), save_path)


def train(
    n_feat = 128,
    lr = 5e-5,
    s = 64,
    m = 0.1,
    n_epoch = 20):

    train_data, train_label, test_data, test_label = load_orl_faces("./orl_faces", False, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Train in {}.".format(device))
    train_data = train_data.to(device)
    train_label = train_label.to(device)
    test_data = test_data.to(device)
    test_label = test_label.to(device)
    train_facedata = FaceDataset(train_data, train_label)
    
    #model = FaceNet_ResNet18(n_feat, train_label.size(0), s=s, m = m, training_mode=training_mode).to(device)
    model = FaceNet_SimpleCNN3(n_feat, train_label.size(0), s = s, m = m, training_mode=training_mode).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    train_loop(train_facedata, model, optimizer, n_epoch, logdir="./runs/SimpleCNN3", save_path="facenet_simplecnn3.pt")

def evaluate(n_feat = 128, feat_extractor_weight = "./facenet_resnet18.pt"):
    train_data, train_label, test_data, test_label = load_orl_faces("./orl_faces", False, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("model inference in {}.".format(device))
    train_data = train_data.to(device)
    train_label = train_label.to(device)
    test_data = test_data.to(device)
    test_label = test_label.to(device)

    #model = FaceNet_ResNet18(n_feat, 0, training_mode=False)
    model = FaceNet_ResNet18_Pretrained()
    #model = FaceNet_SimpleCNN1(n_feat, 0, training_mode=False)
    #model = FaceNet_SimpleCNN2(n_feat, 0, training_mode=False)
    #model = FaceNet_SimpleCNN3(n_feat, 0, training_mode=False)
    #model.load_feat_extractor_state_dict(torch.load(feat_extractor_weight))
    model.thresh = 0.99
    model.eval()
    model = model.to(device)

    eva = Evaluator(model, test_data, test_label)
    with torch.no_grad():
        FAR, FRR, _ = eva.evaluate(False)
    print(FAR, FRR)



if __name__ == '__main__':
    n_feat = 128
    lr = 5e-5
    s = 64
    m = 0.1
    n_epoch = 20
    
    training_mode = False
    evaluate_mode = True

    torch.manual_seed(5014)

    if training_mode:
        train(n_feat = n_feat, lr = lr, s = s, m = m, n_epoch = n_epoch)
    if evaluate_mode:
        evaluate(n_feat, "./facenet_simplecnn3.pt")