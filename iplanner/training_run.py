# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import tqdm
import time
import torch
import json
import wandb
import random
import argparse
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as transforms

from planner_net import PlannerNet
from dataloader import PlannerData
from torch.utils.data import DataLoader
from torchutil import EarlyStopScheduler
from traj_cost import TrajCost
from traj_viz import TrajViz

torch.set_default_dtype(torch.float32)

class PlannerNetTrainer:
    def __init__(self):
        self.root_folder = os.getenv('EXPERIMENT_DIRECTORY', os.getcwd())
        self.load_config()
        self.parse_args()
        self.prepare_model()
        self.prepare_data()
        if self.args.training == True:
            self.init_wandb()
        else:
            print("Testing Mode")
        
    def init_wandb(self):
        # Convert to string in the format you prefer
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        # using Wandb Core
        wandb.require("core")
        # Initialize wandb
        self.wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            project="imperative-path-planning",
            # Set the run name to current date and time
            name=date_time_str + "adamW",
            config={
                "learning_rate": self.args.lr,
                "architecture": "PlannerNet",  # Replace with your actual architecture
                "dataset": self.args.data_root,  # Assuming this holds the dataset name
                "epochs": self.args.epochs,
                "goal_step": self.args.goal_step,
                "max_episode": self.args.max_episode,
                "fear_ahead_dist": self.args.fear_ahead_dist,
            }
        )

    def load_config(self):
        with open(os.path.join(os.path.dirname(self.root_folder), 'config', 'training_config.json')) as json_file:
            self.config = json.load(json_file)

    def prepare_model(self):
        self.net = PlannerNet(self.args.in_channel, self.args.knodes)
        if self.args.resume == True or not self.args.training:
            self.net, self.best_loss = torch.load(self.args.model_save, map_location=torch.device("cpu"))
            print("Resume training from best loss: {}".format(self.best_loss))
        else:
            self.best_loss = float('Inf')

        if torch.cuda.is_available():
            print("Available GPU list: {}".format(list(range(torch.cuda.device_count()))))
            print("Runnin on GPU: {}".format(self.args.gpu_id))
            self.net = self.net.cuda(self.args.gpu_id)

        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
        self.scheduler = EarlyStopScheduler(self.optimizer, factor=self.args.factor, verbose=True, min_lr=self.args.min_lr, patience=self.args.patience)

    def prepare_data(self):
        ids_path = os.path.join(self.args.data_root, self.args.env_id)
        with open(ids_path) as f:
            self.env_list = [line.rstrip() for line in f.readlines()]

        depth_transform = transforms.Compose([
            transforms.Resize((self.args.crop_size)),
            transforms.ToTensor()])
        
        total_img_data = 0
        track_id = 0
        test_env_id = min(self.args.test_env_id, len(self.env_list)-1)
        
        self.train_loader_list = []
        self.val_loader_list   = []
        self.traj_cost_list    = []
        self.traj_viz_list     = []
        
        for env_name in tqdm.tqdm(self.env_list):
            if not self.args.training and track_id != test_env_id:
                track_id += 1
                continue
            is_anymal_frame = False
            sensorOffsetX = 0.0
            camera_tilt = 0.0
            if 'anymal' in env_name:
                is_anymal_frame = True
                sensorOffsetX = self.args.sensor_offsetX_ANYmal
                camera_tilt = self.args.camera_tilt
            elif 'tilt' in env_name:
                camera_tilt = self.args.camera_tilt
            data_path = os.path.join(*[self.args.data_root, self.args.env_type, env_name])

            train_data = PlannerData(root=data_path,
                                     train=True, 
                                     transform=depth_transform,
                                     sensorOffsetX=sensorOffsetX,
                                     is_robot=is_anymal_frame,
                                     goal_step=self.args.goal_step,
                                     max_episode=self.args.max_episode,
                                     max_depth=self.args.max_camera_depth)
            
            total_img_data += len(train_data)
            train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
            self.train_loader_list.append(train_loader)

            val_data = PlannerData(root=data_path,
                                   train=False,
                                   transform=depth_transform,
                                   sensorOffsetX=sensorOffsetX,
                                   is_robot=is_anymal_frame,
                                   goal_step=self.args.goal_step,
                                   max_episode=self.args.max_episode,
                                   max_depth=self.args.max_camera_depth)

            val_loader = DataLoader(val_data, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
            self.val_loader_list.append(val_loader)

            # Load Map and Trajectory Class
            map_name = "tsdf1"
            traj_cost = TrajCost(self.args.gpu_id)
            traj_cost.SetMap(data_path, map_name)

            self.traj_cost_list.append(traj_cost)
            self.traj_viz_list.append(TrajViz(data_path, map_name=map_name, cameraTilt=camera_tilt))
            track_id += 1
            
        print("Data Loading Completed!")
        print("Number of image: %d | Number of goal-image pairs: %d"%(total_img_data, total_img_data * (int)(self.args.max_episode / self.args.goal_step)))
        
        return None

    def MapObsLoss(self, preds, fear, traj_cost, odom, goal, step=0.1):
        waypoints = traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=step)
        loss1, fear_labels = traj_cost.CostofTraj(waypoints, odom, goal, ahead_dist=self.args.fear_ahead_dist)
        loss2 = F.binary_cross_entropy(fear, fear_labels)
        return loss1+loss2, waypoints
    
    def train_epoch(self, epoch):
        loss_sum = 0.0
        env_num = len(self.train_loader_list)
        
        # Zip the lists and convert to a list of tuples
        combined = list(zip(self.train_loader_list, self.traj_cost_list))
        # Shuffle the combined list
        random.shuffle(combined)

        # Iterate through shuffled pairs
        for env_id, (loader, traj_cost) in enumerate(combined):
            train_loss, batches = 0, len(loader)

            enumerater = tqdm.tqdm(enumerate(loader))
            for batch_idx, inputs in enumerater:
                if torch.cuda.is_available():
                    image = inputs[0].cuda(self.args.gpu_id)
                    odom  = inputs[1].cuda(self.args.gpu_id)
                    goal  = inputs[2].cuda(self.args.gpu_id)
                self.optimizer.zero_grad()
                preds, fear = self.net(image, goal)

                loss, _ = self.MapObsLoss(preds, fear, traj_cost, odom, goal)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                enumerater.set_description("Epoch: %d in Env: (%d/%d) - train loss: %.4f on %d/%d" % (epoch, env_id+1, env_num, train_loss/(batch_idx+1), batch_idx, batches))
            
            loss_sum += train_loss/(batch_idx+1)
            wandb.log({"Running Loss": train_loss/(batch_idx+1)})
            
        loss_sum /= env_num

        return loss_sum
        
    def train(self):
        # Convert to string in the format you prefer
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        
        self.args.log_save += (date_time_str + ".txt")
        open(self.args.log_save, 'w').close()

        for epoch in range(self.args.epochs):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(is_visualize=False)
            duration = (time.time() - start_time) / 60 # minutes

            self.log_message("Epoch: %d | Training Loss: %f | Val Loss: %f | Duration: %f" % (epoch, train_loss, val_loss, duration))
            # Log metrics to wandb
            wandb.log({"Avg Training Loss": train_loss, "Validation Loss": val_loss, "Duration (min)": duration})
            
            if val_loss < self.best_loss:
                self.log_message("Save model of epoch %d" % epoch)
                torch.save((self.net, val_loss), self.args.model_save)
                self.best_loss = val_loss
                self.log_message("Current val loss: %.4f" % self.best_loss)
                self.log_message("Epoch: %d model saved | Current Min Val Loss: %f" % (epoch, val_loss))

            self.log_message("------------------------------------------------------------------------")
            if self.scheduler.step(val_loss):
                self.log_message('Early Stopping!')
                break
            
         # Close wandb run at the end of training
        self.wandb_run.finish()
    
    def log_message(self, message):
        with open(self.args.log_save, 'a') as f:
            f.writelines(message)
            f.write('\n')
        print(message)

    def evaluate(self, is_visualize=False):
            self.net.eval()
            test_loss = 0   # Declare and initialize test_loss
            total_batches = 0  # Count total number of batches
            with torch.no_grad():
                for _, (val_loader, traj_cost, traj_viz) in enumerate(zip(self.val_loader_list, self.traj_cost_list, self.traj_viz_list)):
                    preds_viz = []
                    wp_viz = []
                    for batch_idx, inputs in enumerate(val_loader):
                        total_batches += 1  # Increment total number of batches
                        if torch.cuda.is_available():
                            image = inputs[0].cuda(self.args.gpu_id)
                            odom  = inputs[1].cuda(self.args.gpu_id)
                            goal  = inputs[2].cuda(self.args.gpu_id)

                        preds, fear = self.net(image, goal)
                        loss, waypoints = self.MapObsLoss(preds, fear, traj_cost, odom, goal)
                        test_loss += loss.item()

                        if is_visualize and len(preds_viz) < self.args.visual_number:
                            if batch_idx == 0:
                                image_viz = image
                                odom_viz = odom
                                goal_viz = goal
                                fear_viz = fear
                            else:
                                image_viz = torch.cat((image_viz, image), dim=0)
                                odom_viz  = torch.cat((odom_viz, odom),   dim=0)
                                goal_viz  = torch.cat((goal_viz, goal),   dim=0)
                                fear_viz  = torch.cat((fear_viz, fear),   dim=0)
                            preds_viz.extend(preds.tolist())
                            wp_viz.extend(waypoints.tolist())

                    if is_visualize:
                        max_n = min(len(wp_viz), self.args.visual_number)
                        preds_viz = torch.tensor(preds_viz[:max_n])
                        wp_viz    = torch.tensor(wp_viz[:max_n])
                        odom_viz  = odom_viz[:max_n].cpu()
                        goal_viz  = goal_viz[:max_n].cpu()
                        fear_viz  = fear_viz[:max_n, :].cpu()
                        image_viz = image_viz[:max_n].cpu()
                        # visual trajectory and images
                        traj_viz.VizTrajectory(preds_viz, wp_viz, odom_viz, goal_viz, fear_viz)
                        traj_viz.VizImages(preds_viz, wp_viz, odom_viz, goal_viz, fear_viz, image_viz)

                return test_loss / total_batches  # Compute mean test_loss

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Training script for PlannerNet')

        # dataConfig
        parser.add_argument("--data-root", type=str, default=os.path.join(self.root_folder, self.config['dataConfig'].get('data-root')), help="dataset root folder")
        parser.add_argument('--env-id', type=str, default=self.config['dataConfig'].get('env-id'), help='environment id list')
        parser.add_argument('--env_type', type=str, default=self.config['dataConfig'].get('env_type'), help='the dataset type')
        parser.add_argument('--crop-size', nargs='+', type=int, default=self.config['dataConfig'].get('crop-size'), help='image crop size')
        parser.add_argument('--max-camera-depth', type=float, default=self.config['dataConfig'].get('max-camera-depth'), help='maximum depth detection of camera, unit: meter')

        # modelConfig
        parser.add_argument("--model-save", type=str, default=os.path.join(self.root_folder, self.config['modelConfig'].get('model-save')), help="model save point")
        parser.add_argument('--resume', type=str, default=self.config['modelConfig'].get('resume'))
        parser.add_argument('--in-channel', type=int, default=self.config['modelConfig'].get('in-channel'), help='goal input channel numbers')
        parser.add_argument("--knodes", type=int, default=self.config['modelConfig'].get('knodes'), help="number of max nodes predicted")
        parser.add_argument("--goal-step", type=int, default=self.config['modelConfig'].get('goal-step'), help="number of frames betwen goals")
        parser.add_argument("--max-episode", type=int, default=self.config['modelConfig'].get('max-episode-length'), help="maximum episode frame length")

        # trainingConfig
        parser.add_argument('--training', type=str, default=self.config['trainingConfig'].get('training'))
        parser.add_argument("--lr", type=float, default=self.config['trainingConfig'].get('lr'), help="learning rate")
        parser.add_argument("--factor", type=float, default=self.config['trainingConfig'].get('factor'), help="ReduceLROnPlateau factor")
        parser.add_argument("--min-lr", type=float, default=self.config['trainingConfig'].get('min-lr'), help="minimum lr for ReduceLROnPlateau")
        parser.add_argument("--patience", type=int, default=self.config['trainingConfig'].get('patience'), help="patience of epochs for ReduceLROnPlateau")
        parser.add_argument("--epochs", type=int, default=self.config['trainingConfig'].get('epochs'), help="number of training epochs")
        parser.add_argument("--batch-size", type=int, default=self.config['trainingConfig'].get('batch-size'), help="number of minibatch size")
        parser.add_argument("--w-decay", type=float, default=self.config['trainingConfig'].get('w-decay'), help="weight decay of the optimizer")
        parser.add_argument("--num-workers", type=int, default=self.config['trainingConfig'].get('num-workers'), help="number of workers for dataloader")
        parser.add_argument("--gpu-id", type=int, default=self.config['trainingConfig'].get('gpu-id'), help="GPU id")

        # logConfig
        parser.add_argument("--log-save", type=str, default=os.path.join(self.root_folder, self.config['logConfig'].get('log-save')), help="train log file")
        parser.add_argument('--test-env-id', type=int, default=self.config['logConfig'].get('test-env-id'), help='the test env id in the id list')
        parser.add_argument('--visual-number', type=int, default=self.config['logConfig'].get('visual-number'), help='number of visualized trajectories')

        # sensorConfig
        parser.add_argument('--camera-tilt', type=float, default=self.config['sensorConfig'].get('camera-tilt'), help='camera tilt angle for visualization only')
        parser.add_argument('--sensor-offsetX-ANYmal', type=float, default=self.config['sensorConfig'].get('sensor-offsetX-ANYmal'), help='anymal front camera sensor offset in X axis')
        parser.add_argument("--fear-ahead-dist", type=float, default=self.config['sensorConfig'].get('fear-ahead-dist'), help="fear lookahead distance")

        self.args = parser.parse_args()

 
def main():
    trainer = PlannerNetTrainer()
    if trainer.args.training == True:
        trainer.train()
    trainer.evaluate(is_visualize=True)

if __name__ == "__main__":
    main()
