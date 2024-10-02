import numpy as np
import pickle
from sklearn.neighbors import KernelDensity
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.special import entr
import torch
import torch.nn.functional as F
import time
import multiprocessing
from utils.realnvp import RealNVP
import os
import numpy as np
import torch
import random
from collections import deque

class Learn_History_Buffer(object):
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, task):
        self.buffer.append(task)

    def sample(self, batch_size):
        task = self.rng.sample(self.buffer, batch_size)
        return np.vstack(task)

    def __len__(self):
        return len(self.buffer)
    
class StateKernelDensity():
  """
  A KDE-based density model for state items in the replay buffer (e.g., states/goals/state_goal).
  """
  def __init__(self, replay_buffer, batch_size, device="cpu", optimize_every=10, samples=1e5, kernel='gaussian', bandwidth=0.1, normalize=True, 
    log_entropy=False):

    super().__init__()

    self.step = 0
    self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    # self.kde = gaussian_kde(bw_method=0.1)
    self.optimize_every = optimize_every
    self.samples = samples
    self.kernel = kernel
    self.bandwidth = bandwidth
    self.normalize = normalize
    self.kde_sample_mean = 0.
    self.kde_sample_std = 1.
    self.fitted_kde = None
    self.log_entropy = log_entropy
    self.buffer = replay_buffer
    self.device = device
    self.index = np.expand_dims(np.arange(batch_size),1)
    self.scaler = StandardScaler()

  def sample_and_preprocess_batch(self, samples):
      # Extract 
      min_batch = int(1e4)
      n = samples/min_batch
      kde_samples = np.ones((int(samples),4))
      for i in range(int(n)):
        batch = self.buffer.random_batch(min_batch)
        state_batch         = batch["observations"]
        goal_batch          = batch["resampled_goals"]
        kde_samples[i*min_batch:(i+1)*min_batch,:2] = state_batch[:,:2]
        kde_samples[i*min_batch:(i+1)*min_batch,2:] = goal_batch[:,:2]
      return kde_samples

  def _optimize(self, xy_samples=None):
    fit = False
    if self.buffer._size > self.samples/10:
      if xy_samples is not None:
        samples = xy_samples
      else:
        samples = self.sample_and_preprocess_batch(self.samples)


      if self.normalize:
        self.scaler.partial_fit(samples)
        samples = self.scaler.transform(samples)

      self.fitted_kde = self.kde.fit(samples)
      fit = True
    return fit

  def evaluate_log_density(self, samples):
    return self.fitted_kde.score_samples(self.scaler.transform(samples))
    # return self.kde.pdf(self.scaler.transform(samples))

  def evaluate_elementwise_entropy(self, samples, beta=0.):
    log_px = self.fitted_kde.score_samples(self.scaler.transform(samples))
    px = np.exp(log_px)
    elem_entropy = entr(px + beta)
    return elem_entropy

  def task(self, n):
    density = np.exp(self.evaluate_log_density(self.samples_temp[n*self.batch_size:(n+1)*self.batch_size, :]))
    return density

  def save(self, save_folder, t):
    path = os.path.join(save_folder, f"KDE_{t}" + '.joblib')
    joblib.dump(self.kde, path)
    s_path = os.path.join(save_folder, f"scaler_{t}" + '.pkl')
    with open(s_path, 'wb') as f:
        pickle.dump(self.scaler, f)

  def load(self, path, t):
    path = os.path.join(path, f"KDE_{t}" + '.joblib')
    s_path = os.path.join(path, f"scaler_{t}" + '.pkl')
    self.kde = joblib.load(path)
    with open(s_path, 'rb') as f:
        self.scaler = pickle.load(f)
  


class FlowDensity():
  """
  Flow Density model (in this case Real NVP). Similar structure to random density above
  """
  def __init__(self, replay_buffer, batch_size, device, optimize_every=3, train_batch_size=1024, lr=1e-3, num_layer_pairs=3, normalize=True):

    super().__init__()

    self.step = 0
    self.num_layer_pairs = num_layer_pairs
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.train_batch_size = train_batch_size
    self.lazy_load = None
    self.flow_model = None
    self.dev = None
    self.lr = lr
    self.sample_mean = 0.
    self.sample_std = 1.
    self.normalize= normalize
    self.buffer = replay_buffer
    self.device = device
    self.scaler = StandardScaler()

  def _init_from_sample(self, x):
    input_size = x.shape[-1]
    self.input_channel = input_size
    # Device=None is fine for default too based on network.py in realNVP
    self.flow_model = RealNVP(input_channel=self.input_channel, lr=self.lr, num_layer_pairs=self.num_layer_pairs, dev=self.device)

  def evaluate_log_density(self, samples):
    with torch.no_grad():
      log_density = self.flow_model.score_samples(self.scaler.transform(samples))
    return log_density

  def evaluate_density(self, samples):
    subgoal_log_density = self.evaluate_log_density(samples.cpu().detach().numpy())
    row_min = subgoal_log_density.min(axis=-1,keepdims=True)
    row_max = subgoal_log_density.max(-1,keepdims=True)
    norm = (subgoal_log_density - row_min)/(row_max-row_min)
    subgoal_prob = np.exp(norm - norm.max(axis=-1,keepdims=True))/(np.sum(np.exp(norm - norm.max(axis=-1,keepdims=True)),axis=-1,keepdims=True))
    return subgoal_prob
  
  def sample_and_preprocess_batch(self, samples):
    kde_samples = np.ones((int(samples),4))
    batch = self.buffer.random_batch(samples)
    state_batch         = batch["observations"]
    goal_batch          = batch["resampled_goals"]
    kde_samples[:,:2] = state_batch[:,:2]
    kde_samples[:,2:] = goal_batch[:,:2]
    return kde_samples
    
  @property
  def ready(self):
    return self.flow_model is not None

  def _optimize(self, flow_samples=None, image_env=False, encoder=None):
    fit = False
    if self.buffer._size > self.train_batch_size:
      if flow_samples is not None:
        samples = flow_samples
      else:
        flow_samples = self.sample_and_preprocess_batch(self.train_batch_size)
        samples = flow_samples # [:,:2]

      if image_env:
        with torch.no_grad():
          samples = torch.FloatTensor(samples).view(-1, 3, 84, 84).to(self.device)
          samples = encoder(samples).cpu().numpy()

      if self.normalize:
        self.scaler.partial_fit(samples)
        samples = self.scaler.transform(samples)

      # lazy load the model if not yet loaded
      if self.flow_model is None:
        self._init_from_sample(samples)
        if self.lazy_load is not None:
          self.load(self.lazy_load)
          self.lazy_load = None

      samples = torch.FloatTensor(samples)
      #del self.flow_model
      #self.flow_model = RealNVP(input_channel=self.input_channel, lr=self.lr, num_layer_pairs=self.num_layer_pairs, dev=self.dev)
      self.flow_model.fit(samples, epochs=self.optimize_every)
    fit = True
    her_data_density = np.mean(np.exp(self.evaluate_log_density(flow_samples)))
    return fit, her_data_density

  def save(self, save_folder, t):
    path = os.path.join(save_folder, f"flowdensity_{t}" + '.pt')
    if self.flow_model is not None:
      torch.save({
        'flow_model': self.flow_model,
      }, path)
    s_path = os.path.join(save_folder, f"scaler_{t}" + '.pkl')
    with open(s_path, 'wb') as f:
        pickle.dump(self.scaler, f)

  def load(self, save_folder : str, t):
    path = os.path.join(save_folder, self.module_name + '.pt')
    s_path = os.path.join(save_folder, f"scaler_{t}" + '.pkl')
    if self.flow_model is None and os.path.exists(path):
      self.lazy_load = save_folder
    else:
      self.flow_model = torch.load(path)
    with open(s_path, 'rb') as f:
      self.scaler = pickle.load(f)

