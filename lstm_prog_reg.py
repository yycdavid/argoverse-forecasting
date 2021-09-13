from copy import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union
import argparse
import wandb
import time

from prog_utils import *
import utils.baseline_utils as baseline_utils
import utils.baseline_config as config
from utils.lstm_utils import ModelUtils, LSTMDataset
from prog_utils import *
from models import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# some global variable
global_step = 0
best_loss = float("inf")
prev_loss = best_loss
decrement_counter = 0

np.random.seed(100)

def train(
        prep_data: Any,
        trajectory_loader: Any,
        epoch: int,
        criterion: Any,
        encoder: Any,
        cls_encoder: Any,
        combine_net: Any,
        prog_decoder: Any,
        encoder_optimizer: Any,
        cls_encoder_optimizer: Any,
        prog_decoder_optimizer: Any,
        combine_optimizer: Any,
        model_utils: ModelUtils,
        rollout_len: int = 30,
        mode: str = "Train",

) -> None:
    """Train the lstm network.

    Args:
        train_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        encoder_optimizer: optimizer for the encoder network
        model_utils: instance for ModelUtils class
        rollout_len: current prediction horizon
        use_traj: whether use trajectory prediction for training

    """
    args = parse_arguments()
    global global_step
    global best_loss
    global prev_loss
    global decrement_counter

    forecasted_programs = [[] for i in range(len(trajectory_loader.dataset))]
    total_loss = []
    batch_id = 0
    batch_size_all = -1

    if mode == "Train":
        encoder.train()
        cls_encoder.train()
        combine_net.train()
        prog_decoder.train()
    else:
        encoder.eval()
        cls_encoder.eval()
        combine_net.eval()
        prog_decoder.eval()

    for _, (_input, target, helpers) in tqdm(enumerate(trajectory_loader)):

        cls = []
        cls_num = []

        # Pack all centerlines into one tensor
        for j in range(len(helpers)):
            cls_num.append(len(helpers[j][0]))
            for k in range(10):
                # Downsample centerlines by 10
                if k < len(helpers[j][0]):
                    cl_current = torch.FloatTensor(helpers[j][0][k])[::10, :]
                # pad 0-sequence to ensure that each datum has the same number of centerlines
                else:
                    cl_current = torch.zeros((16,2))
                cls.append(cl_current)
        cls_num = torch.LongTensor(cls_num)
        cls = torch.nn.utils.rnn.pad_sequence(cls, batch_first=True)
        # Clever way to generating mask in one line: http://juditacs.github.io/2018/12/27/masked-attention.html
        cls_mask = (torch.arange(10)[None, :] < cls_num[:, None])

        # programs: (batch_size, 3, 3)
        programs = []
        for j in range(len(helpers)):
            programs.append(helpers[j][1])
        programs = torch.tensor(programs)

        _input = _input.to(device)
        cls = cls.to(device)
        programs = programs.to(device)

        if mode == "Train":
            encoder_optimizer.zero_grad()
            cls_encoder_optimizer.zero_grad()
            combine_optimizer.zero_grad()
            prog_decoder_optimizer.zero_grad()

        # Encoder
        batch_size = _input.shape[0]
        if batch_size_all == -1:
            batch_size_all = batch_size
        cls_batch_size = cls.shape[0]
        input_length = _input.shape[1]
        centerline_length = cls.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        cls_encoder_hidden = model_utils.init_hidden(
            cls_batch_size,
            cls_encoder.module.hidden_size if use_cuda else cls_encoder.hidden_size)

        # Encode centerlines
        for ei in range(centerline_length):
            cls_encoder_input = cls[:, ei, :]
            cls_encoder_hidden = cls_encoder(cls_encoder_input, cls_encoder_hidden)
        cls_encoder_hidden = cls_encoder_hidden[0]

        # Reshape centerline tensor
        # (batch_size, 10, length of longest centerline, 2)
        cls = cls.reshape(_input.shape[0], -1, cls.shape[-2], cls.shape[-1])

        # Initialize losses
        loss = 0
        prog_loss = 0
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        # Encode observed trajectory
        for ei in range(_input.shape[1]):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Compute centerline cross-entropy loss
        centerline_scores = combine_net(encoder_hidden[0], cls_encoder_hidden.view(batch_size, 10, 16).view(batch_size, -1))

        # Masking invalid centerline scores
        centerline_scores[~cls_mask] = float('-inf')
        centerline_gt = programs[:, 0, 0].long()
        centerline_loss = ce_loss(centerline_scores, centerline_gt)

        # Choose predicted centerline and its feature
        centerline_pred_index = torch.argmax(centerline_scores, dim=1)
        cls_encoder_hidden = cls_encoder_hidden.reshape(batch_size, -1, cls_encoder_hidden.shape[-1])
        centerline_pred = cls[np.arange(batch_size), centerline_pred_index]

        # Store the feature of the selected centerlines
        cls_encoder_hidden_pred = cls_encoder_hidden[np.arange(batch_size), centerline_pred_index]

        # Decode timestep and velocity
        prog_decoder_features = torch.cat([encoder_hidden[0], cls_encoder_hidden_pred], dim=1)
        prog_output = prog_decoder(prog_decoder_features)
        timestep_pred = prog_output[:, 0]
        velocity_pred = prog_output[:, -1]
        timestep_gt = programs[:, 0, 1]
        velocity_gt = programs[:, 0, 2]

        # Aggregate all losses
        timestep_loss = mse_loss(timestep_pred, timestep_gt)
        velocity_loss = mse_loss(velocity_pred, velocity_gt)
        prog_loss += centerline_loss + timestep_loss / 100 + velocity_loss * 50

        batch_id += 1
        loss = prog_loss
        if mode != "Train":
            total_loss.append(loss)

        if mode == "Train":
            # Backpropagate
            loss.backward()

            encoder_optimizer.step()
            cls_encoder_optimizer.step()
            combine_optimizer.step()
            prog_decoder_optimizer.step()

            if global_step % 100 == 0:
                # Log results
                print(
                    f"Train -- Epoch:{epoch}, Step:{global_step}, Loss:{loss}, Timestep: {timestep_loss:.2f}, Centerline: {centerline_loss:.2f}, Velocity: {velocity_loss:.2f}")

                # logger.scalar_summary(tag="Train/loss",
                #                       value=loss.item(),
                #                       step=epoch)
                save_dir = "saved_models/lstm_prog_reg_train"
                os.makedirs(save_dir, exist_ok=True)
                model_utils.save_checkpoint(
                    save_dir,
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "rollout_len": rollout_len,
                        "encoder_state_dict": encoder.state_dict(),
                        "cls_encoder_state_dict": cls_encoder.state_dict(),
                        "combine_net_state_dict": combine_net.state_dict(),
                        "prog_decoder_state_dict": prog_decoder.state_dict(),
                        "best_loss": loss,
                        "encoder_optimizer": encoder_optimizer.state_dict(),
                        "cls_encoder_optimizer": cls_encoder_optimizer.state_dict(),
                        "prog_decoder_optimizer": prog_decoder_optimizer.state_dict()
                    },
                )
            global_step += 1
            if args.wandb:
                wandb.log({f'{mode}/loss': loss.item(),
                            f'{mode}/centerline_loss': centerline_loss.item(),
                            f'{mode}/velocity_loss': velocity_loss.item(),
                            f'{mode}/timestep_loss': timestep_loss.item()})
    if mode == "Val":
        val_loss = sum(total_loss) / len(total_loss)
        if val_loss <= best_loss:
            best_loss = val_loss

            save_dir = "saved_models/lstm_prog_reg_val"
            os.makedirs(save_dir, exist_ok=True)
            model_utils.save_checkpoint(
                save_dir,
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "rollout_len": rollout_len,
                    "encoder_state_dict": encoder.state_dict(),
                    "cls_encoder_state_dict": cls_encoder.state_dict(),
                    "combine_net_state_dict": combine_net.state_dict(),
                    "prog_decoder_state_dict": prog_decoder.state_dict(),
                    "best_loss": val_loss,
                    "encoder_optimizer": encoder_optimizer.state_dict(),
                    "cls_encoder_optimizer": cls_encoder_optimizer.state_dict(),
                    "prog_decoder_optimizer": prog_decoder_optimizer.state_dict()
                },
            )
        if val_loss <= prev_loss:
            decrement_counter = 0
        else:
            decrement_counter += 1
        if args.wandb:
            wandb.log({f'{mode}/loss': loss.item(),
                        f'{mode}/centerline_loss': centerline_loss.item(),
                        f'{mode}/velocity_loss': velocity_loss.item(),
                        f'{mode}/timestep_loss': timestep_loss.item()})

def main_split(args):
    """
    Split the data into one segment per training instance
    """
    data = pd.read_pickle(args.data_path)
    splitted = []
    print("Before split, {} instances".format(data.shape[0]))
    for i in tqdm(range(data.shape[0])):
        row = data.loc[i]
        features = row.FEATURES[:args.obs_len]
        for j in range(len(row.PROG)):
            splitted.append([row.SEQUENCE, features, row.CANDIDATE_CENTERLINES, [row.PROG[j]]])
            xy = get_xy(features)
            cl, n_step, v = row.PROG[j]
            next_seg = exec_prog(xy, row.CANDIDATE_CENTERLINES[cl], n_step, v)
            for p in next_seg:
                new_step = copy(features[-1])
                new_step[0] = str(float(new_step[0]) + 0.1)
                new_step[3:5] = p
                features = np.append(features, np.expand_dims(new_step, 0), axis=0)

    result = pd.concat([pd.DataFrame([entry], columns=data.columns) for entry in splitted], ignore_index=True)

    print("After split, {} instances".format(result.shape[0]))
    ss = args.data_path[:-4]
    new_path = ss+'_split.pkl'
    result.to_pickle(new_path)

def main_train(args):
    if use_cuda:
        print(f"Using all ({torch.cuda.device_count()}) GPUs...")

    model_utils = ModelUtils()

    new_data_train = pd.read_pickle(args.train_path)
    new_data_val = pd.read_pickle(args.val_path)

    data_dict = baseline_utils.get_reg_prog_data(new_data_train, new_data_val, args)

    # Get model
    criterion = nn.MSELoss()
    encoder = EncoderRNN(
        input_size=len(baseline_utils.BASELINE_INPUT_FEATURES['none']))
    cls_encoder = EncoderRNN(input_size=2)
    # TODO: remove hardcoding of input and output sizes
    prog_decoder = ProgDecoder(input_size=32, hidden_size=128, output_size=2)
    combine_net = CombineNet(input_size=16*(10+1), hidden_size=256, output_size=10)
    if use_cuda:
        encoder = nn.DataParallel(encoder)
        cls_encoder = nn.DataParallel(cls_encoder)
        prog_decoder = nn.DataParallel(prog_decoder)
        combine_net = nn.DataParallel(combine_net)

    encoder.to(device)
    cls_encoder.to(device)
    prog_decoder.to(device)
    combine_net.to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    cls_encoder_optimizer = torch.optim.Adam(cls_encoder.parameters(), lr=args.lr)
    combine_optimizer = torch.optim.Adam(combine_net.parameters(), lr=args.lr)
    prog_decoder_optimizer = torch.optim.Adam(prog_decoder.parameters(), lr=args.lr)

    # If model_path provided, resume from saved checkpoint
    if args.model_path is not None and os.path.isfile(args.model_path):
        epoch, rollout_len, _ = model_utils.load_checkpoint(
            args.model_path, encoder, cls_encoder, combine_net, prog_decoder, encoder_optimizer, cls_encoder_optimizer, combine_optimizer, prog_decoder_optimizer)
        print("{} model loaded!".format(args.model_path))
        start_epoch = epoch + 1

    else:
        start_epoch = 0

    if not args.test:
        # TODO: starthere, change to output manager
        if args.wandb:
            wandb.init(project='traj',
                        group='lstm-prog',
                        config=args)
        # Tensorboard logger
        log_dir = os.path.join(os.getcwd(), "lstm_logs", 'none')

        # Get PyTorch Dataset and Dataloader
        val_dataset = LSTMDataset(data_dict, args, "val")
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=model_utils.my_collate_fn,
        )

        train_dataset = LSTMDataset(data_dict, args, "train")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=model_utils.my_collate_fn,
        )

        print("Training begins ...")

        global global_step
        global best_loss
        global prev_loss
        global decrement_counter

        epoch = start_epoch
        global_start_time = time.time()
        logger = None
        best_loss = float("inf")
        prev_loss = best_loss
        while epoch < args.end_epoch:
            start = time.time()
            train(
                None,
                train_loader,
                epoch,
                criterion,
                encoder,
                cls_encoder,
                combine_net,
                prog_decoder,
                encoder_optimizer,
                cls_encoder_optimizer,
                prog_decoder_optimizer,
                combine_optimizer,
                model_utils,
                30, # disable rollout_length curriculum
                mode="Train"
            )
            end = time.time()

            print(
                f"Training epoch completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
            )
            epoch += 1
            with torch.no_grad():
                if epoch % 3 == 0:
                    start = time.time()
                    train(
                        None,
                        val_loader,
                        epoch,
                        criterion,
                        encoder,
                        cls_encoder,
                        combine_net,
                        prog_decoder,
                        encoder_optimizer,
                        cls_encoder_optimizer,
                        prog_decoder_optimizer,
                        combine_optimizer,
                        model_utils,
                        30,
                        mode="Val"
                    )
                    end = time.time()
                    print(
                        f"Validation completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                    )

                    # If val loss increased 3 times consecutively, go to next rollout length
                    if decrement_counter > 2:
                        break

def main():
    args = parse_arguments()
    if args.mode == 'split':
        main_split(args)
    else:
        main_train(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
