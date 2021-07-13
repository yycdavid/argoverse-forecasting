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
import numpy as np
import pandas as pd 
import pickle as pkl
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger
import utils.baseline_config as config
import utils.baseline_utils as baseline_utils
from utils.lstm_utils import ModelUtils, LSTMDataset
from prog_utils import * 

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
global_step = 0
best_loss = float("inf")
np.random.seed(100)

ROLLOUT_LENS = [1, 10, 30]


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=1,
                        help="Test batch size")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position",
    )
    parser.add_argument(
        "--train_features",
        default="forecasting_features_train.pkl",
        type=str,
        help="path to the file which has train features.",
    )
    parser.add_argument(
        "--val_features",
        default="forecasting_features_val.pkl",
        type=str,
        help="path to the file which has val features.",
    )
    parser.add_argument(
        "--test_features",
        default="forecasting_features_test.pkl",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--use_map",
                        action="store_true",
                        help="Use the map based features")
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")
    parser.add_argument("--test",
                        action="store_true",
                        help="If true, only run the inference")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=1,
                        help="Training batch size")
    parser.add_argument("--val_batch_size",
                        type=int,
                        default=1,
                        help="Val batch size")
    parser.add_argument("--end_epoch",
                        type=int,
                        default=5000,
                        help="Last epoch")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument(
        "--traj_save_path",
        required=False,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )
    return parser.parse_args()


class EncoderRNN(nn.Module):
    """Encoder Network."""
    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 16):
        """Initialize the encoder network.

        Args:
            input_size: number of features in the input
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM

        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        return hidden


class DecoderRNN(nn.Module):
    """Decoder Network."""
    def __init__(self, embedding_size=8, hidden_size=16, output_size=2):
        """Initialize the decoder network.

        Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output

        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            output: output from lstm
            hidden: final hidden state

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden


class ProgDecoder(nn.Module):
    """Program Decoder Network."""
    def __init__(self, input_size=16, hidden_size=64, output_size=30+15+15+2):
        """Initialize the decoder network.

        Args:
            input_size: 
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output

        """
        super(ProgDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            output: output from lstm
            hidden: final hidden state

        """
        hidden = F.relu(self.linear1(x))
        output = self.linear2(hidden)
        return output 


def train(
        train_loader: Any,
        epoch: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        cls_encoder: Any, 
        decoder: Any,
        prog_decoder: Any, 
        encoder_optimizer: Any,
        cls_encoder_optimizer: Any, 
        decoder_optimizer: Any,
        prog_decoder_optimizer: Any, 
        model_utils: ModelUtils,
        rollout_len: int = 30,
        use_traj: bool = False, 
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

    for i, (_input, target, helpers) in enumerate(train_loader):

        cls = []
        cls_num = [0]

        # Pack all centerlines into one tensor
        for j in range(len(helpers)):
            cls_num.append(len(helpers[j][2])-cls_num[-1])
            for k in range(10):
                # Downsample centerlines by 10
                if k < len(helpers[j][2]):
                    cl_current = torch.FloatTensor(helpers[j][2][k])[::10, :]
                # pad 0-sequence to ensure that each datum has the same number of centerlines
                else:
                    cl_current = torch.zeros((16,2))
                cls.append(cl_current)

        cls_num = cls_num[1:]
        cls = torch.nn.utils.rnn.pad_sequence(cls, batch_first=True)

        # programs: (batch_size, 3, 3)
        programs = []
        for j in range(len(helpers)):
            programs.append(helpers[j][9])
        programs = torch.tensor(programs)

        _input = _input.to(device)
        cls = cls.to(device)
        target = target.to(device)
        programs = programs.to(device)

        # Set to train mode
        encoder.train()
        cls_encoder.train()
        decoder.train()
        prog_decoder.train()

        # Zero the gradients
        if use_traj:
            decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        cls_encoder_optimizer.zero_grad()
        prog_decoder_optimizer.zero_grad()

        # Encoder
        batch_size = _input.shape[0]
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

        # Initialize losses
        loss = 0
        traj_loss = 0
        prog_loss = 0 
        ce_loss = nn.CrossEntropyLoss() 
        mse_loss = nn.MSELoss()

        # Iteratively decode centerlines and compute losses
        for t in range(programs.shape[1]):

            # Encode observed trajectory
            for ei in range(_input.shape[1]):
                encoder_input = _input[:, ei, :]
                encoder_hidden = encoder(encoder_input, encoder_hidden)

            # Compute centerline cross-entropy loss
            centerline_scores = torch.softmax(torch.matmul(encoder_hidden[0], cls_encoder_hidden.view(-1, cls_encoder_hidden.shape[-1]).transpose(1,0)), dim=1)
            centerline_gt = programs[:, t, 0].long()
            centerline_loss = ce_loss(centerline_scores, centerline_gt)

            # Choose predicted centerline and its feature
            centerline_pred_index = torch.argmax(centerline_scores, dim=1)
            cls_encoder_hidden = cls_encoder_hidden.reshape(batch_size, -1, cls_encoder_hidden.shape[-1])
            centerline_pred = cls[np.arange(batch_size), centerline_pred_index]
            cls_encoder_hidden_pred = cls_encoder_hidden[np.arange(batch_size), centerline_pred_index]

            # Decode timestep and velocity
            prog_decoder_features = torch.cat([encoder_hidden[0], cls_encoder_hidden_pred], dim=1)
            prog_output = prog_decoder(prog_decoder_features)
            timestep_scores = prog_output[:, :-1]
            timestep_pred = torch.argmax(timestep_scores, dim=1)
            velocity_pred = prog_output[:, -1]

            timestep_gt = programs[:, t, 1].long()
            velocity_gt = programs[:, t, 2]
            # print(timestep_gt, timestep_scores)
            # Aggregate all losses
            timestep_loss = ce_loss(timestep_scores, timestep_gt)
            velocity_loss = mse_loss(velocity_pred, velocity_gt)
            prog_loss += centerline_loss + timestep_loss + velocity_loss

            # Decode programs into trajectories and treat them as new inputs
            # TODO: can we batch-ify this step?

            new_input = []
            for i in range(_input.shape[0]):
                new_input_i = exec_prog(_input[i].cpu().numpy(), centerline_pred[i].cpu().numpy(), timestep_pred[i].cpu().numpy(), velocity_pred[i].detach().cpu().numpy())
                new_input_i = np.array(new_input_i)
                new_input.append(new_input_i)
            _input = torch.FloatTensor(new_input).to(device)

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

        if use_traj:
            loss = loss + prog_loss 
        else:
            loss = prog_loss 

        # # Get average loss for pred_len
        # loss = loss / rollout_len

        # Backpropagate
        loss.backward()

        encoder_optimizer.step()
        cls_encoder_optimizer.step()
        prog_decoder_optimizer.step()
        if use_traj:
            decoder_optimizer.step()


        if global_step % 1000 == 0:

            # Log results
            print(
                f"Train -- Epoch:{epoch}, loss:{loss}, Rollout:{rollout_len}")

            logger.scalar_summary(tag="Train/loss",
                                  value=loss.item(),
                                  step=epoch)

        global_step += 1


def validate(
        val_loader: Any,
        epoch: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        cls_encoder: Any, 
        decoder: Any,
        prog_decoder: Any, 
        encoder_optimizer: Any,
        cls_encoder_optimizer: Any, 
        decoder_optimizer: Any,
        prog_decoder_optimizer: Any, 
        model_utils: ModelUtils,
        prev_loss: float,
        decrement_counter: int,
        rollout_len: int = 30,
        use_traj: bool = False 
) -> Tuple[float, int]:
    """Validate the lstm network.

    Args:
        val_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        prev_loss: Loss in the previous validation run
        decrement_counter: keeping track of the number of consecutive times loss increased in the current rollout
        rollout_len: current prediction horizon

    """
    args = parse_arguments()
    global best_loss
    total_loss = []

    for i, (_input, target, helpers) in enumerate(val_loader):

        cls = []
        cls_num = [0]

        # Pack all centerlines into one tensor
        for j in range(len(helpers)):
            cls_num.append(len(helpers[j][2])-cls_num[-1])
            for k in range(10):
                # Downsample centerlines by 10
                if k < len(helpers[j][2]):
                    cl_current = torch.FloatTensor(helpers[j][2][k])[::10, :]
                # pad 0-sequence to ensure that each datum has the same number of centerlines
                else:
                    cl_current = torch.zeros((16,2))
                cls.append(cl_current)

        cls_num = cls_num[1:]
        cls = torch.nn.utils.rnn.pad_sequence(cls, batch_first=True)

        # programs: (batch_size, 3, 3)
        programs = []
        for j in range(len(helpers)):
            programs.append(helpers[j][9])
        programs = torch.tensor(programs)

        _input = _input.to(device)
        cls = cls.to(device)
        target = target.to(device)
        programs = programs.to(device)

        # Set to train mode
        encoder.train()
        cls_encoder.train()
        decoder.train()
        prog_decoder.train()

        # Zero the gradients
        if use_traj:
            decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        cls_encoder_optimizer.zero_grad()
        prog_decoder_optimizer.zero_grad()

        # Encoder
        batch_size = _input.shape[0]
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

        # Initialize losses
        loss = 0
        traj_loss = 0
        prog_loss = 0 
        ce_loss = nn.CrossEntropyLoss() 
        mse_loss = nn.MSELoss()

        # Iteratively decode centerlines and compute losses
        for t in range(programs.shape[1]):

            # Encode observed trajectory
            for ei in range(_input.shape[1]):
                encoder_input = _input[:, ei, :]
                encoder_hidden = encoder(encoder_input, encoder_hidden)

            # Compute centerline cross-entropy loss
            centerline_scores = torch.softmax(torch.matmul(encoder_hidden[0], cls_encoder_hidden.view(-1, cls_encoder_hidden.shape[-1]).transpose(1,0)), dim=1)
            centerline_gt = programs[:, t, 0].long()
            centerline_loss = ce_loss(centerline_scores, centerline_gt)

            # Choose predicted centerline and its feature
            centerline_pred_index = torch.argmax(centerline_scores, dim=1)
            cls_encoder_hidden = cls_encoder_hidden.reshape(batch_size, -1, cls_encoder_hidden.shape[-1])
            centerline_pred = cls[np.arange(batch_size), centerline_pred_index]
            cls_encoder_hidden_pred = cls_encoder_hidden[np.arange(batch_size), centerline_pred_index]

            # Decode timestep and velocity
            prog_decoder_features = torch.cat([encoder_hidden[0], cls_encoder_hidden_pred], dim=1)
            prog_output = prog_decoder(prog_decoder_features)
            timestep_scores = prog_output[:, :-1]
            timestep_pred = torch.argmax(timestep_scores, dim=1)
            velocity_pred = prog_output[:, -1]

            timestep_gt = programs[:, t, 1].long()
            velocity_gt = programs[:, t, 2]
            # print(timestep_gt, timestep_scores)
            # Aggregate all losses
            timestep_loss = ce_loss(timestep_scores, timestep_gt)
            velocity_loss = mse_loss(velocity_pred, velocity_gt)
            prog_loss += centerline_loss + timestep_loss + velocity_loss

            # Decode programs into trajectories and treat them as new inputs
            # TODO: can we batch-ify this step?

            new_input = []
            for i in range(_input.shape[0]):
                new_input_i = exec_prog(_input[i].cpu().numpy(), centerline_pred[i].cpu().numpy(), timestep_pred[i].cpu().numpy(), velocity_pred[i].detach().cpu().numpy())
                new_input_i = np.array(new_input_i)
                new_input.append(new_input_i)
            _input = torch.FloatTensor(new_input).to(device)
            
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

        if use_traj:
            loss = loss + prog_loss 
        else:
            loss = prog_loss

        # # Get average loss for pred_len
        # loss = loss / output_length
        total_loss.append(loss)

        if i % 10 == 0:

            cprint(
                f"Val -- Epoch:{epoch}, loss:{loss}, Rollout: {rollout_len}",
                color="green",
            )

    # Save
    val_loss = sum(total_loss) / len(total_loss)

    if val_loss <= best_loss:
        best_loss = val_loss
        if args.use_map:
            save_dir = "saved_models/lstm_prog_map"
        elif args.use_social:
            save_dir = "saved_models/lstm_prog_social"
        else:
            save_dir = "saved_models/lstm_prog"

        os.makedirs(save_dir, exist_ok=True)
        model_utils.save_checkpoint(
            save_dir,
            {
                "epoch": epoch + 1,
                "rollout_len": rollout_len,
                "encoder_state_dict": encoder.state_dict(),
                "cls_encoder_state_dict": cls_encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "prog_decoder_state_dict": prog_decoder.state_dict(), 
                "best_loss": val_loss,
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "cls_encoder_optimizer": cls_encoder_optimizer.state_dict(), 
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "prog_decoder_optimizer": prog_decoder_optimizer.state_dict()
            },
        )

    logger.scalar_summary(tag="Val/loss", value=val_loss.item(), step=epoch)

    # Keep track of the loss to change preiction horizon
    if val_loss <= prev_loss:
        decrement_counter = 0
    else:
        decrement_counter += 1

    return val_loss, decrement_counter


def infer_absolute(
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
):
    """Infer function for non-map LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance

    """
    args = parse_arguments()
    forecasted_trajectories = {}

    for i, (_input, target, helpers) in enumerate(test_loader):

        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(
            (batch_size, args.pred_len, 2)).to(device)

        # Decode hidden state in future trajectory
        for di in range(args.pred_len):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get absolute trajectory
        abs_helpers = {}
        abs_helpers["REFERENCE"] = np.array(helpers_dict["DELTA_REFERENCE"])
        abs_helpers["TRANSLATION"] = np.array(helpers_dict["TRANSLATION"])
        abs_helpers["ROTATION"] = np.array(helpers_dict["ROTATION"])
        abs_inputs, abs_outputs = baseline_utils.get_abs_traj(
            _input.clone().cpu().numpy(),
            decoder_outputs.detach().clone().cpu().numpy(),
            args,
            abs_helpers,
        )

        for i in range(abs_outputs.shape[0]):
            seq_id = int(helpers_dict["SEQ_PATHS"][i])
            forecasted_trajectories[seq_id] = [abs_outputs[i]]

    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)

        
def infer_map(
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
):
    """Infer function for map-based LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance

    """
    args = parse_arguments()
    global best_loss
    forecasted_trajectories = {}
    for i, (_input, target, helpers) in enumerate(test_loader):

        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]

        # Iterate over every element in the batch
        for batch_idx in range(batch_size):
            num_candidates = len(
                helpers_dict["CANDIDATE_CENTERLINES"][batch_idx])
            curr_centroids = helpers_dict["CENTROIDS"][batch_idx]
            seq_id = int(helpers_dict["SEQ_PATHS"][batch_idx])
            abs_outputs = []

            # Predict using every centerline candidate for the current trajectory
            for candidate_idx in range(num_candidates):
                curr_centerline = helpers_dict["CANDIDATE_CENTERLINES"][
                    batch_idx][candidate_idx]
                curr_nt_dist = helpers_dict["CANDIDATE_NT_DISTANCES"][
                    batch_idx][candidate_idx]

                _input = torch.FloatTensor(
                    np.expand_dims(curr_nt_dist[:args.obs_len].astype(float),
                                   0)).to(device)

                # Initialize encoder hidden state
                encoder_hidden = model_utils.init_hidden(
                    1, encoder.module.hidden_size
                    if use_cuda else encoder.hidden_size)

                # Encode observed trajectory
                for ei in range(input_length):
                    encoder_input = _input[:, ei, :]
                    encoder_hidden = encoder(encoder_input, encoder_hidden)

                # Initialize decoder input with last coordinate in encoder
                decoder_input = encoder_input[:, :2]

                # Initialize decoder hidden state as encoder hidden state
                decoder_hidden = encoder_hidden

                decoder_outputs = torch.zeros((1, args.pred_len, 2)).to(device)

                # Decode hidden state in future trajectory
                for di in range(args.pred_len):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    decoder_outputs[:, di, :] = decoder_output

                    # Use own predictions as inputs at next step
                    decoder_input = decoder_output

                # Get absolute trajectory
                abs_helpers = {}
                abs_helpers["REFERENCE"] = np.expand_dims(
                    np.array(helpers_dict["CANDIDATE_DELTA_REFERENCES"]
                             [batch_idx][candidate_idx]),
                    0,
                )
                abs_helpers["CENTERLINE"] = np.expand_dims(curr_centerline, 0)

                abs_input, abs_output = baseline_utils.get_abs_traj(
                    _input.clone().cpu().numpy(),
                    decoder_outputs.detach().clone().cpu().numpy(),
                    args,
                    abs_helpers,
                )

                # array of shape (1,30,2) to list of (30,2)
                abs_outputs.append(abs_output[0])
            forecasted_trajectories[seq_id] = abs_outputs

    os.makedirs(forecasted_save_dir, exist_ok=True)
    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)


def infer_helper(
        curr_data_dict: Dict[str, Any],
        start_idx: int,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        model_utils: ModelUtils,
        forecasted_save_dir: str,
):
    """Run inference on the current joblib batch.

    Args:
        curr_data_dict: Data dictionary for the current joblib batch
        start_idx: Start idx of the current joblib batch
        encoder: Encoder network instance
        decoder: Decoder network instance
        model_utils: ModelUtils instance
        forecasted_save_dir: Directory where forecasted trajectories are to be saved

    """
    args = parse_arguments()
    curr_test_dataset = LSTMDataset(curr_data_dict, args, "test")
    curr_test_loader = torch.utils.data.DataLoader(
        curr_test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        collate_fn=model_utils.my_collate_fn,
    )

    if args.use_map:
        print(f"#### LSTM+map inference at index {start_idx} ####")
        infer_map(
            curr_test_loader,
            encoder,
            decoder,
            start_idx,
            forecasted_save_dir,
            model_utils,
        )

    else:
        print(f"#### LSTM+social inference at {start_idx} ####"
              ) if args.use_social else print(
                  f"#### LSTM inference at {start_idx} ####")
        infer_absolute(
            curr_test_loader,
            encoder,
            decoder,
            start_idx,
            forecasted_save_dir,
            model_utils,
        )


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
    new_data = pd.read_pickle('train_prep_para.pkl')
    data_dict['train_helpers'].CANDIDATE_CENTERLINES = new_data.CANDIDATE_CENTERLINES
    data_dict['train_helpers']['PROG'] = new_data.PROG
    data_dict['val_helpers'].CANDIDATE_CENTERLINES = new_data.CANDIDATE_CENTERLINES
    data_dict['val_helpers']['PROG'] = new_data.PROG 

    # Get model
    criterion = nn.MSELoss()
    encoder = EncoderRNN(
        input_size=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]))
    cls_encoder = EncoderRNN(input_size=2)
    decoder = DecoderRNN(output_size=2)
    # TODO: remove hardcoding of input and output sizes
    prog_decoder = ProgDecoder(input_size=32, hidden_size=128, output_size=32)
    if use_cuda:
        encoder = nn.DataParallel(encoder)
        cls_encoder = nn.DataParallel(cls_encoder)
        decoder = nn.DataParallel(decoder)
        prog_decoder = nn.DataParallel(prog_decoder)
    encoder.to(device)
    cls_encoder.to(device)
    decoder.to(device)
    prog_decoder.to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    cls_encoder_optimizer = torch.optim.Adam(cls_encoder.parameters(), lr = args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    prog_decoder_optimizer = torch.optim.Adam(prog_decoder.parameters(), lr=args.lr)

    # If model_path provided, resume from saved checkpoint
    if args.model_path is not None and os.path.isfile(args.model_path):
        epoch, rollout_len, _ = model_utils.load_checkpoint(
            args.model_path, encoder, decoder, encoder_optimizer,
            decoder_optimizer)
        start_epoch = epoch + 1
        start_rollout_idx = ROLLOUT_LENS.index(rollout_len) + 1

    else:
        start_epoch = 0
        start_rollout_idx = 0

    if not args.test:

        # Tensorboard logger
        log_dir = os.path.join(os.getcwd(), "lstm_logs", baseline_key)

        # Get PyTorch Dataset
        val_dataset = LSTMDataset(data_dict, args, "val")
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=model_utils.my_collate_fn,
        )

        train_dataset = LSTMDataset(data_dict, args, "train")

        # Setting Dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=model_utils.my_collate_fn,
        )



        print("Training begins ...")

        decrement_counter = 0

        epoch = start_epoch
        global_start_time = time.time()
        for i in range(start_rollout_idx, len(ROLLOUT_LENS)):
            rollout_len = ROLLOUT_LENS[i]
            logger = Logger(log_dir, name="{}".format(rollout_len))
            best_loss = float("inf")
            prev_loss = best_loss
            while epoch < args.end_epoch:
                start = time.time()
                train(
                    train_loader,
                    epoch,
                    criterion,
                    logger,
                    encoder,
                    cls_encoder,
                    decoder,
                    prog_decoder, 
                    encoder_optimizer,
                    cls_encoder_optimizer,
                    decoder_optimizer,
                    prog_decoder_optimizer, 
                    model_utils,
                    30, # disable rollout_length curriculum 
                    False, # do not use traj prediction for training
                )
                end = time.time()

                print(
                    f"Training epoch completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                )

                epoch += 1
                if epoch % 5 == 0:
                    start = time.time()
                    prev_loss, decrement_counter = validate(
                        val_loader,
                        epoch,
                        criterion,
                        logger,
                        encoder,
                        cls_encoder,
                        decoder,
                        prog_decoder, 
                        encoder_optimizer,
                        cls_encoder_optimizer,
                        decoder_optimizer,
                        prog_decoder_optimizer,
                        model_utils,
                        prev_loss,
                        decrement_counter,
                        rollout_len,
                        False,
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

        # test_batch_size should be lesser than joblib_batch_size
        Parallel(n_jobs=-2, verbose=2)(
            delayed(infer_helper)(test_data_subsets[i], i, encoder, decoder,
                                  model_utils, temp_save_dir)
            for i in range(0, test_size, args.joblib_batch_size))

        baseline_utils.merge_saved_traj(temp_save_dir, args.traj_save_path)
        shutil.rmtree(temp_save_dir)

        end = time.time()
        print(f"Test completed in {(end - start_time) / 60.0} mins")
        print(f"Forecasted Trajectories saved at {args.traj_save_path}")


if __name__ == "__main__":
    main()
