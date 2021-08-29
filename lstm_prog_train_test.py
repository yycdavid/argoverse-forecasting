"""lstm_train_test.py runs the LSTM baselines training/inference on forecasting dataset.

Note: The training code for these baselines is covered under the patent <PATENT_LINK>.

Example usage:
python lstm_train_test.py 
    --model_path saved_models/lstm.pth.tar 
    --test_features ../data/forecasting_data_test.pkl 
    --train_features ../data/forecasting_data_train.pkl 
    --val_features ../data/forecasting_data_val.pkl 
    --use_delta --normalize
"""

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union

import argparse
import joblib
from joblib import Parallel, delayed
import multiprocessing as mp 
import numpy as np
import pandas as pd 
import pickle as pkl
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
import wandb 

# from logger import Logger
import utils.baseline_config as config
import utils.baseline_utils as baseline_utils
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

ROLLOUT_LENS = [1, 10, 30]
PROGRAM_LENS = [1, 2, 3]


def train(
        prep_data: Any, 
        trajectory_loader: Any,
        epoch: int,
        criterion: Any,
        encoder: Any,
        cls_encoder: Any, 
        combine_net: Any, 
        decoder: Any,
        prog_decoder: Any, 
        encoder_optimizer: Any,
        cls_encoder_optimizer: Any, 
        decoder_optimizer: Any,
        prog_decoder_optimizer: Any, 
        combine_optimizer: Any, 
        model_utils: ModelUtils,
        rollout_len: int = 30,
        use_traj: bool = False, 
        total_segments: int = 1,
        mode: str = "Train",
        
) -> None:
    """Train the lstm network.

    Args:
        train_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
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
        decoder.train()
        prog_decoder.train()
    else:
        encoder.eval()
        cls_encoder.eval()
        combine_net.eval()
        decoder.eval()
        prog_decoder.eval()

    for _, (_input, target, helpers) in enumerate(trajectory_loader):

        cls = []
        cls_num = []

        # Pack all centerlines into one tensor
        for j in range(len(helpers)):
            cls_num.append(len(helpers[j][2]))
            for k in range(10):
                # Downsample centerlines by 10
                if k < len(helpers[j][2]):
                    cl_current = torch.FloatTensor(helpers[j][2][k])[::10, :]
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
            programs.append(helpers[j][9])
        programs = torch.tensor(programs)

        _input = _input.to(device)
        cls = cls.to(device)
        target = target.to(device)
        programs = programs.to(device)

        if mode == "Train":
            if use_traj:
                decoder_optimizer.zero_grad()
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
        output_length = target.shape[1]
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
        traj_loss = 0
        prog_loss = 0 
        ce_loss = nn.CrossEntropyLoss() 
        mse_loss = nn.MSELoss()

        # Iteratively decode centerlines and compute losses
        for t in range(total_segments):

            # Encode observed trajectory
            for ei in range(_input.shape[1]):
                encoder_input = _input[:, ei, :]
                encoder_hidden = encoder(encoder_input, encoder_hidden)

            # Compute centerline cross-entropy loss
            centerline_scores = combine_net(encoder_hidden[0], cls_encoder_hidden.view(batch_size, 10, 16).view(batch_size, -1))

            # Masking invalid centerline scores
            centerline_scores[~cls_mask] = float('-inf')
            centerline_gt = programs[:, t, 0].long()
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
            # timestep_pred = torch.nn.Sigmoid()(prog_output[:, 0]) * 30
            timestep_pred = prog_output[:, 0]
            
            # timestep_scores = prog_output[:, :-1]
            # timestep_pred = torch.argmax(timestep_scores, dim=1)
            # if total_segments != 1:
            #     timestep_pred = torch.nn.Sigmoid()(prog_output[:, 0]) * 30
            # else:
            #     timestep_pred = prog_output[:, 0]
            velocity_pred = prog_output[:, -1]

            timestep_gt = programs[:, t, 1]
            velocity_gt = programs[:, t, 2]

            # Aggregate all losses
            # timestep_loss = ce_loss(timestep_scores, timestep_gt)
            timestep_loss = mse_loss(timestep_pred, timestep_gt)
            velocity_loss = mse_loss(velocity_pred, velocity_gt)
            prog_loss += centerline_loss + timestep_loss / 100 + velocity_loss * 50

            if total_segments != 1:
                new_input = []
                for i in range(_input.shape[0]):
                    new_input_i = exec_prog(_input[i].cpu().numpy(), centerline_pred[i].cpu().numpy(),
                    timestep_pred[i].detach().cpu().numpy().round().astype(int), velocity_pred[i].detach().cpu().numpy())
                    new_input_i = torch.FloatTensor(new_input_i)
                    new_input.append(new_input_i)

                    # Store the predicted program
                    if mode == "Test":
                        forecasted_programs[batch_id * batch_size_all + i].append((
                        int(centerline_pred_index[i].cpu()),
                        int(round(float(timestep_pred[i].long().cpu()))),
                        float(velocity_pred[i].detach().cpu())))

                _input = torch.nn.utils.rnn.pad_sequence(new_input, batch_first=True).to(device)

                # Deal with the special case of timestep = 0
                if len(_input.size()) == 2:
                    _input = _input.unsqueeze(0) 

            # Decode trajectories if used for training
            if use_traj and t == 0:
                # Initialize decoder input with last coordinate in encoder
                decoder_input = encoder_input[:, :2]

                # Initialize decoder hidden state as encoder hidden state
                decoder_hidden = encoder_hidden
                decoder_outputs = torch.zeros(target.shape).to(device)

                # Decode hidden state in future trajectory
                for di in range(rollout_len):
                    decoder_output, decoder_hidden = decoder(decoder_input,
                                                            decoder_hidden)
                    decoder_outputs[:, di, :] = decoder_output

                    # Update loss
                    traj_loss += criterion(decoder_output[:, :2], target[:, di, :2])

                    # Use own predictions as inputs at next step
                    decoder_input = decoder_output
        batch_id += 1

        if use_traj:
            loss = traj_loss + prog_loss 
        else:
            loss = prog_loss 

        # # Get average loss for pred_len
        # loss = loss / rollout_len
        if mode != "Train":
            total_loss.append(loss) 

        if mode == "Train":
            # Backpropagate
            loss.backward()

            encoder_optimizer.step()
            cls_encoder_optimizer.step()
            combine_optimizer.step() 
            prog_decoder_optimizer.step()
            if use_traj:
                decoder_optimizer.step()

            if global_step % 100 == 0:
                # Log results
                print(
                    f"Train -- Epoch:{epoch}, Step:{global_step}, Loss:{loss}, Timestep: {timestep_loss:.2f}, Centerline: {centerline_loss:.2f}, Velocity: {velocity_loss:.2f}")

                # logger.scalar_summary(tag="Train/loss",
                #                       value=loss.item(),
                #                       step=epoch)
                save_dir = "saved_models/lstm_prog_{}segment_train".format(total_segments)
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
                        "decoder_state_dict": decoder.state_dict(),
                        "prog_decoder_state_dict": prog_decoder.state_dict(), 
                        "best_loss": loss,
                        "encoder_optimizer": encoder_optimizer.state_dict(),
                        "cls_encoder_optimizer": cls_encoder_optimizer.state_dict(), 
                        "decoder_optimizer": decoder_optimizer.state_dict(),
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

            save_dir = "saved_models/lstm_prog_{}segment_val".format(total_segments)
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
                    "decoder_state_dict": decoder.state_dict(),
                    "prog_decoder_state_dict": prog_decoder.state_dict(), 
                    "best_loss": val_loss,
                    "encoder_optimizer": encoder_optimizer.state_dict(),
                    "cls_encoder_optimizer": cls_encoder_optimizer.state_dict(), 
                    "decoder_optimizer": decoder_optimizer.state_dict(),
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
    if mode == "Test":
        forecasted_save_dir = 'Traj'
        prep_data['PROG_PRED'] = forecasted_programs
        os.makedirs(forecasted_save_dir, exist_ok=True)
        with open(os.path.join(forecasted_save_dir, f"test_prep_{total_segments}_seg_new.pkl"),
                "wb") as f:
            pkl.dump(prep_data, f)

def infer_program(
        prep_data: Any, 
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        cls_encoder: Any,
        combine_net: Any,  
        decoder: DecoderRNN,
        prog_decoder: Any, 
        model_utils: ModelUtils,
        forecasted_save_dir: str,
        total_segments: int = 1
):
    """Infer function for program-based LSTM baselines.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance

    """
    args = parse_arguments()
    forecasted_programs = [[] for i in range(len(test_loader.dataset))]

    batch_id = 0
    batch_size_all = -1
    correct = 0
    for _, (_input, target, helpers) in tqdm(enumerate(test_loader)):

        cls = []
        cls_num = []

        # Pack all centerlines into one tensor
        for j in range(len(helpers)):
            cls_num.append(len(helpers[j][2]))
            for k in range(10):
                # Downsample centerlines by 10
                if k < len(helpers[j][2]):
                    cl_current = torch.FloatTensor(helpers[j][2][k])[::10, :]
                # pad 0-sequence to ensure that each datum has the same number of centerlines
                else:
                    cl_current = torch.zeros((16,2))
                cls.append(cl_current)

        cls_num = torch.LongTensor(cls_num)
        cls = torch.nn.utils.rnn.pad_sequence(cls, batch_first=True)
        cls_mask = (torch.arange(10)[None, :] < cls_num[:, None])

        # programs: (batch_size, 3, 3)
        programs = []
        for j in range(len(helpers)):
            programs.append(helpers[j][9])
        programs = torch.tensor(programs)

        _input = _input.to(device)
        cls = cls.to(device)
        target = target.to(device)
        programs = programs.to(device)

        # Set to eval mode
        encoder.eval()
        cls_encoder.eval()
        decoder.eval()
        prog_decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        if batch_size_all == -1:
            batch_size_all = batch_size
        cls_batch_size = cls.shape[0]
        input_length = _input.shape[1]
        centerline_length = cls.shape[1]
        output_length = target.shape[1]
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
        cls = cls.reshape(_input.shape[0], -1, cls.shape[-2], cls.shape[-1])

        ce_loss = nn.CrossEntropyLoss() 
        mse_loss = nn.MSELoss()

        for t in range(total_segments):

            # Encode observed trajectory
            for ei in range(_input.shape[1]):
                encoder_input = _input[:, ei, :]
                encoder_hidden = encoder(encoder_input, encoder_hidden)

            centerline_scores = combine_net(encoder_hidden[0], cls_encoder_hidden.view(batch_size, 10, 16).view(batch_size, -1))
            centerline_scores[~cls_mask] = float('-inf')
            centerline_pred_index = torch.argmax(centerline_scores, dim=1)
            cls_encoder_hidden = cls_encoder_hidden.reshape(batch_size, -1, cls_encoder_hidden.shape[-1])
            centerline_pred = cls[np.arange(batch_size), centerline_pred_index]

            cls_encoder_hidden_pred = cls_encoder_hidden[np.arange(batch_size), centerline_pred_index]

            # Decode timestep and velocity
            prog_decoder_features = torch.cat([encoder_hidden[0], cls_encoder_hidden_pred], dim=1)
            prog_output = prog_decoder(prog_decoder_features)
            if total_segments != 1:
                timestep_pred = torch.nn.Sigmoid()(prog_output[:, 0]) * 30
            else:
                timestep_pred = prog_output[:, 0]
            velocity_pred = prog_output[:, -1]

            # Decode programs into trajectories and treat them as new inputs
            # TODO: can we batch-ify this step?
            new_input = []
            for i in range(_input.shape[0]):
                new_input_i = exec_prog(_input[i].cpu().numpy(), centerline_pred[i].cpu().numpy(),
                timestep_pred[i].detach().cpu().numpy().round().astype(int), velocity_pred[i].detach().cpu().numpy())
                new_input_i = torch.FloatTensor(new_input_i)
                new_input.append(new_input_i)
                forecasted_programs[batch_id * batch_size_all + i].append((
                    int(centerline_pred_index[i].cpu()),
                    int(round(float(timestep_pred[i].long().cpu()))),
                    float(velocity_pred[i].detach().cpu())))

            # _input = torch.FloatTensor(new_input).to(device)
            _input = torch.nn.utils.rnn.pad_sequence(new_input, batch_first=True).to(device)

            # Deal with the special case of timestep = 0
            if len(_input.size()) == 2:
                _input = _input.unsqueeze(0) 
        batch_id += 1

    prep_data['PROG_PRED'] = forecasted_programs
    os.makedirs(forecasted_save_dir, exist_ok=True)
    with open(os.path.join(forecasted_save_dir, f"test_prep_{total_segments}_seg_new.pkl"),
              "wb") as f:
        pkl.dump(prep_data, f)

def main():
    """Main."""
    args = parse_arguments()

    if not baseline_utils.validate_args(args):
        return

    print(f"Using all ({joblib.cpu_count()}) CPUs....")
    if use_cuda:
        print(f"Using all ({torch.cuda.device_count()}) GPUs...")

    model_utils = ModelUtils()

    # key for getting feature set
    # Get features
    if args.use_map and args.use_social:
        baseline_key = "map_social"
    elif args.use_map:
        baseline_key = "map"
    elif args.use_social:
        baseline_key = "social"
    else:
        baseline_key = "none"

    # Get data
    data_dict = baseline_utils.get_data(args, baseline_key)

    # add the program data to the data_dict
    if args.total_segments == 3:
        new_data_train = pd.read_pickle('Traj/train_prep.pkl')
        new_data_val = pd.read_pickle('Traj/val_prep.pkl')
        test_prep = pd.read_pickle('Traj/val_prep.pkl')
        if args.regularized:
            new_data_train = pd.read_pickle('Traj/train_pen_2.pkl')
            new_data_val = pd.read_pickle('Traj/val_pen_2.pkl')
            test_prep = pd.read_pickle('Traj/val_pen_2.pkl')
    elif args.total_segments == 1:
        new_data_train = pd.read_pickle('Traj/train_1_seg.pkl')
        new_data_val = pd.read_pickle('Traj/val_1_seg.pkl')
        test_prep = pd.read_pickle('Traj/val_1_seg.pkl')

    # data_dict['train_helpers'].CANDIDATE_CENTERLINES = new_data_train.CANDIDATE_CENTERLINES
    # data_dict['val_helpers'].CANDIDATE_CENTERLINES = new_data_val.CANDIDATE_CENTERLINES

    data_dict['train_helpers'].CANDIDATE_CENTERLINES = new_data_train.CLS_RELATIVE
    data_dict['train_helpers']['PROG'] = new_data_train.PROG
    data_dict['val_helpers'].CANDIDATE_CENTERLINES = new_data_val.CLS_RELATIVE
    data_dict['val_helpers']['PROG'] = new_data_val.PROG 
    data_dict['test_helpers'].CANDIDATE_CENTERLINES = new_data_val.CLS_RELATIVE
    data_dict['test_helpers']['PROG'] = new_data_val.PROG 

    # Get model
    criterion = nn.MSELoss()
    encoder = EncoderRNN(
        input_size=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]))
    cls_encoder = EncoderRNN(input_size=2)
    decoder = DecoderRNN(output_size=2)
    # TODO: remove hardcoding of input and output sizes
    prog_decoder = ProgDecoder(input_size=32, hidden_size=128, output_size=2)
    combine_net = CombineNet(input_size=16*(10+1), hidden_size=256, output_size=10)
    if use_cuda:
        encoder = nn.DataParallel(encoder)
        cls_encoder = nn.DataParallel(cls_encoder)
        decoder = nn.DataParallel(decoder)
        prog_decoder = nn.DataParallel(prog_decoder)
        combine_net = nn.DataParallel(combine_net)

    encoder.to(device)
    cls_encoder.to(device)
    decoder.to(device)
    prog_decoder.to(device)
    combine_net.to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    cls_encoder_optimizer = torch.optim.Adam(cls_encoder.parameters(), lr=args.lr)
    combine_optimizer = torch.optim.Adam(combine_net.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    prog_decoder_optimizer = torch.optim.Adam(prog_decoder.parameters(), lr=args.lr)

    # If model_path provided, resume from saved checkpoint
    if args.model_path is not None and os.path.isfile(args.model_path):
        epoch, rollout_len, _ = model_utils.load_checkpoint(
            args.model_path, encoder, cls_encoder, combine_net,
            decoder, prog_decoder,
             encoder_optimizer, cls_encoder_optimizer, combine_optimizer,
             decoder_optimizer,
              prog_decoder_optimizer)
        print("{} model loaded!".format(args.model_path))
        start_epoch = epoch + 1
        start_rollout_idx = PROGRAM_LENS.index(rollout_len) + 1

    else:
        start_epoch = 0
        start_rollout_idx = 0

    if not args.test:
        if args.wandb:
            wandb.init(project='traj',
                        group='lstm-prog',
                        config=args)
        # Tensorboard logger
        log_dir = os.path.join(os.getcwd(), "lstm_logs", baseline_key)

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
        for i in range(start_rollout_idx, len(PROGRAM_LENS)):
            program_len = PROGRAM_LENS[i]
            # logger = Logger(log_dir, name="{}".format(rollout_len))
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
                    decoder,
                    prog_decoder, 
                    encoder_optimizer,
                    cls_encoder_optimizer,
                    decoder_optimizer,
                    prog_decoder_optimizer, 
                    combine_optimizer,
                    model_utils,
                    30, # disable rollout_length curriculum 
                    False, # do not use traj prediction for training
                    total_segments=program_len,
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
                            decoder,
                            prog_decoder, 
                            encoder_optimizer,
                            cls_encoder_optimizer,
                            decoder_optimizer,
                            prog_decoder_optimizer,
                            combine_optimizer, 
                            model_utils,
                            30,
                            False,
                            total_segments=program_len,
                            mode="Val"
                        )
                        end = time.time()
                        print(
                            f"Validation completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                        )

                        # If val loss increased 3 times consecutively, go to next rollout length
                        if decrement_counter > 2:
                            break

    else:
        start_time = time.time()
        temp_save_dir = tempfile.mkdtemp()

        test_size = data_dict["test_input"].shape[0]
        test_data_subsets = baseline_utils.get_test_data_dict_subset(
            data_dict, args)

        # Get PyTorch Dataset and Dataloader
        test_dataset = LSTMDataset(data_dict, args, "val")
        # test_dataset = LSTMDataset(data_dict, args, "test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.val_batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=model_utils.my_collate_fn,
        )

        with torch.no_grad():
            start = time.time()
            train(
                None,
                test_loader,
                epoch,
                criterion,
                encoder,
                cls_encoder,
                decoder,
                combine_net, 
                prog_decoder, 
                encoder_optimizer,
                cls_encoder_optimizer,
                decoder_optimizer,
                prog_decoder_optimizer,
                combine_optimizer, 
                model_utils,
                rollout_len=30,
                use_traj=False,
                total_segments=3, # output the whole program
                mode="Test"
            )
            end = time.time()

        print(f"Test completed in {(end - start_time) / 60.0} mins")
        # print(f"Forecasted Trajectories saved at {args.traj_save_path}")


if __name__ == "__main__":
    main()
