# ConvNet.py (PyTorch version)

import os
import datetime
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import DataIO


class ConvNet(nn.Module):
    """
    PyTorch implementation of the CNN denoiser used in the Iterative BP-CNN system.

    Input:  numpy array of shape (batch_size, feature_length)
    Output: numpy array of shape (batch_size, label_length)
    """

    def __init__(self, net_config_in, train_config_in, net_id, device=None):
        super(ConvNet, self).__init__()

        self.net_config = net_config_in
        self.train_config = train_config_in
        self.net_id = net_id

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # For loading residual noise properties later
        self.res_noise_power_dict = {}
        self.res_noise_pdf_dict = {}

        # Build conv1d layers to mimic original tf.nn.conv2d pipeline
        layers = []
        in_channels = 1
        for layer in range(self.net_config.total_layers):
            out_channels = int(self.net_config.feature_map_nums[layer])
            ksize = int(self.net_config.filter_sizes[layer])

            # "SAME" padding approximation
            padding = ksize // 2

            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ksize,
                padding=padding,
            )
            layers.append(conv)

            if layer != self.net_config.total_layers - 1:
                layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels

        self.net = nn.Sequential(*layers).to(self.device)

    # ------------------------------------------------------------------ #
    # Forward / inference helpers
    # ------------------------------------------------------------------ #

    def forward(self, x):
        """
        x: torch.Tensor or numpy array of shape (batch_size, feature_length)
        returns: torch.Tensor of shape (batch_size, label_length)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))

        # Shape: (batch, 1, feature_length)
        x = x.to(self.device)
        x = x.unsqueeze(1)
        y = self.net(x)  # (batch, out_channels, feature_length)

        # Last layer has feature_map_nums[-1] == 1, so squeeze channel
        y = y.squeeze(1)
        return y

    def infer_numpy(self, x_np):
        """
        Convenience: run forward pass on numpy input and return numpy output.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x_np)
        return out.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Saving / restoring (replaces TF Saver)
    # ------------------------------------------------------------------ #

    def _model_folder_from_id(self, model_id):
        # model(a_b_c) convention kept from original code
        model_id_str = np.array2string(
            model_id[0 : (self.net_id + 1)],
            separator="_",
            formatter={"int": lambda d: "%d" % d},
        )
        model_id_str = model_id_str[1 : (len(model_id_str) - 1)]
        save_model_folder = "%snetid%d_model%s" % (
            self.net_config.model_folder,
            self.net_id,
            model_id_str,
        )
        return save_model_folder

    def save_network(self, model_id):
        folder = self._model_folder_from_id(model_id)
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = os.path.join(folder, "model.pt")
        torch.save(self.state_dict(), save_path)
        print("Saved PyTorch model to %s" % save_path)

    def restore_network_with_model_id(self, model_id):
        folder = self._model_folder_from_id(model_id)
        load_path = os.path.join(folder, "model.pt")
        if not os.path.exists(load_path):
            print("Warning: model file %s not found, starting from scratch." % load_path)
            return
        state = torch.load(load_path, map_location=self.device)
        self.load_state_dict(state)
        print("Restored PyTorch model from %s" % load_path)

    # ------------------------------------------------------------------ #
    # Normality test (PyTorch version of the original TF code)
    # ------------------------------------------------------------------ #

    def calc_normality_test_torch(
        self, residual_noise: torch.Tensor, batch_size: int, batch_size_for_norm_test: int
    ):
        """
        residual_noise: (batch_size, label_length)
        """
        groups = int(batch_size // batch_size_for_norm_test)
        # reshape to (groups, label_len * batch_size_for_norm_test)
        x = residual_noise.view(groups, -1)

        mean = x.mean(dim=1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        moment_3rd = ((x - mean) ** 3).mean(dim=1, keepdim=True)
        moment_4th = ((x - mean) ** 4).mean(dim=1, keepdim=True)

        skewness = moment_3rd / (variance.pow(1.5) + 1e-10)
        kurtosis = moment_4th / (variance.pow(2.0) + 1e-10)

        norm_test = torch.mean(skewness.pow(2) + 0.25 * (kurtosis - 3.0).pow(2))
        return norm_test

    # ------------------------------------------------------------------ #
    # Train / test (replaces TF session-run loop)
    # ------------------------------------------------------------------ #

    def _compute_loss(self, pred, target, batch_size):
        """
        pred, target: (batch_size, label_length)
        Returns scalar loss tensor.
        """
        mse = torch.mean((pred - target) ** 2)

        if not self.train_config.normality_test_enabled:
            return mse

        norm_test = self.calc_normality_test_torch(
            residual_noise=target - pred,
            batch_size=batch_size,
            batch_size_for_norm_test=1,
        )

        if np.isinf(self.train_config.normality_lambda):
            # Only normality test influences shape, but original paper uses
            # MSE + lambda * normality, so here we keep MSE for stability.
            return mse
        else:
            return mse + self.train_config.normality_lambda * norm_test

    def test_network_online(self, dataio):
        """
        Evaluate average loss on the test set.
        """
        self.eval()
        remain_samples = self.train_config.test_sample_num
        batch_size_min = self.train_config.test_minibatch_size

        total_loss = 0.0
        with torch.no_grad():
            while remain_samples > 0:
                load_batch_size = min(batch_size_min, remain_samples)

                batch_xs, batch_ys = dataio.load_batch_for_test(load_batch_size)
                xs = torch.from_numpy(batch_xs.astype(np.float32)).to(self.device)
                ys = torch.from_numpy(batch_ys.astype(np.float32)).to(self.device)

                y_pred = self.forward(xs)
                loss = self._compute_loss(y_pred, ys, load_batch_size)

                total_loss += loss.item() * load_batch_size
                remain_samples -= load_batch_size

        avg_loss = total_loss / float(self.train_config.test_sample_num)
        print("Test loss: %.6f" % avg_loss)
        self.train()
        return avg_loss

    def train_network(self, model_id):
        """
        Full training loop; mirrors the original TF logic but in PyTorch.
        """
        if self.train_config is None:
            raise RuntimeError("train_config is None; ConvNet created for inference only.")

        start = datetime.datetime.now()

        # Data loaders
        dataio_train = DataIO.TrainingDataIO(
            self.train_config.training_feature_file,
            self.train_config.training_label_file,
            self.train_config.training_sample_num,
            self.net_config.feature_length,
            self.net_config.label_length,
        )
        dataio_test = DataIO.TestDataIO(
            self.train_config.test_feature_file,
            self.train_config.test_label_file,
            self.train_config.test_sample_num,
            self.net_config.feature_length,
            self.net_config.label_length,
        )

        # Restore if previous model exists
        self.restore_network_with_model_id(model_id)

        optimizer = optim.Adam(self.parameters())
        self.to(self.device)
        self.train()

        print("Initial evaluation on test set...")
        best_loss = self.test_network_online(dataio_test)
        best_state = copy.deepcopy(self.state_dict())
        no_improve_count = 0

        print("Start training...")
        for epoch in range(1, self.train_config.epoch_num + 1):
            batch_xs, batch_ys = dataio_train.load_next_mini_batch(
                self.train_config.training_minibatch_size
            )
            xs = torch.from_numpy(batch_xs.astype(np.float32)).to(self.device)
            ys = torch.from_numpy(batch_ys.astype(np.float32)).to(self.device)

            optimizer.zero_grad()
            y_pred = self.forward(xs)
            loss = self._compute_loss(y_pred, ys, self.train_config.training_minibatch_size)
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0 or epoch == self.train_config.epoch_num:
                print("Epoch %d, train loss: %.6f" % (epoch, loss.item()))
                val_loss = self.test_network_online(dataio_test)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(self.state_dict())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= 8:
                        print("Early stopping (no improvement for 8 evaluations).")
                        break

        # Load best weights, save model
        self.load_state_dict(best_state)
        self.save_network(model_id)

        end = datetime.datetime.now()
        print("Final minimum loss: %f" % best_loss)
        print("Used time for training: %ds" % (end - start).seconds)

    # ------------------------------------------------------------------ #
    # Residual noise helpers (unchanged logic, just Python/numpy)
    # ------------------------------------------------------------------ #

    def get_res_noise_power(self, model_id, SNRset=np.zeros(0)):
        """
        Load residual noise power from precomputed text file.
        """
        if self.res_noise_power_dict.__len__() == 0:
            model_id_str = np.array2string(
                model_id[0 : (self.net_id + 1)],
                separator="_",
                formatter={"int": lambda d: "%d" % d},
            )
            model_id_str = model_id_str[1 : (len(model_id_str) - 1)]
            residual_noise_power_file = "%sresidual_noise_property_netid%d_model%s.txt" % (
                self.net_config.residual_noise_property_folder,
                self.net_id,
                model_id_str,
            )

            data = np.loadtxt(residual_noise_power_file, dtype=np.float32)
            shape_data = np.shape(data)
            if np.size(shape_data) == 1:
                self.res_noise_power_dict[data[0]] = data[1 : shape_data[0]]
            else:
                SNR_num = shape_data[0]
                for i in range(SNR_num):
                    self.res_noise_power_dict[data[i, 0]] = data[i, 1 : shape_data[1]]
        return self.res_noise_power_dict

    def get_res_noise_pdf(self, model_id):
        """
        Load residual noise empirical PDF from text file.
        """
        if self.res_noise_pdf_dict.__len__() == 0:
            model_id_str = np.array2string(
                model_id[0 : (self.net_id + 1)],
                separator="_",
                formatter={"int": lambda d: "%d" % d},
            )
            model_id_str = model_id_str[1 : (len(model_id_str) - 1)]
            residual_noise_pdf_file = "%sresidual_noise_property_netid%d_model%s.txt" % (
                self.net_config.residual_noise_property_folder,
                self.net_id,
                model_id_str,
            )
            data = np.loadtxt(residual_noise_pdf_file, dtype=np.float32)
            shape_data = np.shape(data)
            if np.size(shape_data) == 1:
                self.res_noise_pdf_dict[data[0]] = data[1 : shape_data[0]]
            else:
                SNR_num = shape_data[0]
                for i in range(SNR_num):
                    self.res_noise_pdf_dict[data[i, 0]] = data[i, 1 : shape_data[1]]
        return self.res_noise_pdf_dict