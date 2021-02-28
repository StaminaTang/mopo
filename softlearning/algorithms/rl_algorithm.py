import abc
from collections import OrderedDict
from itertools import count
import gtimer as gt
import math
import os
import pdb

import tensorflow as tf
import numpy as np

from softlearning.samplers import rollouts
from softlearning.misc.utils import save_video

#算法 训练与评估实现
class RLAlgorithm(tf.contrib.checkpoint.Checkpointable):
    """Abstract RLAlgorithm.

    Implements the _train and _methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            n_epochs=1000,
            train_every_n_steps=1,
            n_train_repeat=1,
            max_train_repeat_per_timestep=5,
            n_initial_exploration_steps=0,
            initial_exploration_policy=None,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render_mode=None,
            video_save_frequency=0,
            session=None,
            **kwargs
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.训练数
            n_train_repeat (`int`): Number of times to repeat the training每个时间步训练数
                for single time step.
            n_initial_exploration_steps: Number of steps in the beginning to
                take using actions drawn from a separate exploration policy.初始化探索策略
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.   用来评估的rollouts数目 
            eval_deterministic (`int`): Whether or not to run the policy in在评估策略的时候是否使用deterministic
                deterministic mode when evaluating policy.
            eval_render_mode (`str`): Mode to render evaluation rollouts in.
                None to disable rendering.
        """
        self.sampler = sampler
#sf里的强化学习算法。采样、epoch数、训练数目、每个时间步训练数、epoch长度、探索策略数目、探索策略、评估数目、评估是否使用deterministic、视频频率保存
        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._max_train_repeat_per_timestep = max(
            max_train_repeat_per_timestep, n_train_repeat)
        self._train_every_n_steps = train_every_n_steps
        self._epoch_length = epoch_length
        self._n_initial_exploration_steps = n_initial_exploration_steps
        self._initial_exploration_policy = initial_exploration_policy

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._video_save_frequency = video_save_frequency

        if self._video_save_frequency > 0:
            assert eval_render_mode != 'human', (
                "RlAlgorithm cannot render and save videos at the same time")
            self._eval_render_mode = 'rgb_array'
        else:
            self._eval_render_mode = eval_render_mode

        self._session = session or tf.keras.backend.get_session()

        self._epoch = 0
        self._timestep = 0
        self._num_train_steps = 0
#初始化结束
#定义初始化探索hook
    def _initial_exploration_hook(self, env, initial_exploration_policy, pool):
        if self._n_initial_exploration_steps < 1: return
#如果没有提供初始化探索策略
        if not initial_exploration_policy:
            raise ValueError(
                "Initial exploration policy must be provided when"
                " n_initial_exploration_steps > 0.")
#已经提供初始化策略
#首先采样的初始化：
        self.sampler.initialize(env, initial_exploration_policy, pool)
        while pool.size < self._n_initial_exploration_steps:
            self.sampler.sample()
#定义在训练之前
    def _training_before_hook(self):
        """Method called before the actual training loops."""
        pass
#定义在训练之后
    def _training_after_hook(self):
        """Method called after the actual training loops."""
        pass
#定义在每个时间步开始时被调用的hook
    def _timestep_before_hook(self, *args, **kwargs):
        """Hook called at the beginning of each timestep."""
        pass
#定义在每个时间步结束时被调用的hook
    def _timestep_after_hook(self, *args, **kwargs):
        """Hook called at the end of each timestep."""
        pass
#在每个epoch开始时被调用的hook
    def _epoch_before_hook(self):
        """Hook called at the beginning of each epoch."""
        self._train_steps_this_epoch = 0
#在每个epoch结束时被调用的hook
    def _epoch_after_hook(self, *args, **kwargs):
        """Hook called at the end of each epoch."""
        pass
#训练批：return的是对随机批的采样
    def _training_batch(self, batch_size=None):
        return self.sampler.random_batch(batch_size)
#评估批：return的是训练批
    def _evaluation_batch(self, *args, **kwargs):
        return self._training_batch(*args, **kwargs)
#定义：训练开始
    @property
    def _training_started(self):
        return self._total_timestep > 0
#定义整个时间步
    @property
    def _total_timestep(self):
        total_timestep = self._epoch * self._epoch_length + self._timestep
        return total_timestep
#定义训练函数
    def _train(self):
        """Return a generator that performs RL training.
#返回一个experimentrunner实现Rl训练
#设置Rl训练参数环境：env为了训练的环境；策略：为了训练的策略；初始化探索策略，如果设为无的话那么连探索也用的是同一个策略；
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        training_environment = self._training_environment#训练环境
        evaluation_environment = self._evaluation_environment#评估环境
        policy = self._policy#策略
        pool = self._pool#缓冲区
#如果训练还没有开始，那么先开始初始化训练，初始化探索hook。
        if not self._training_started:
            self._init_training()

            self._initial_exploration_hook(
                training_environment, self._initial_exploration_policy, pool)
#开始采样的初始化
        self.sampler.initialize(training_environment, policy, pool)
#gtimer开始，重置root，重命名root为RLAlgorithm，设置定义unique为False
        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)
#设置training之前的hook
        self._training_before_hook()
#所有环境的采样=把所有缓冲区里的采样传给环境
        env_samples = pool.return_all_samples()
#开始循环，训练开始的epochs
        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')
#循环里开始采样，时间步里面的采样。
            start_samples = self.sampler._total_samples
            for i in count():
                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                if (samples_now >= start_samples + self._epoch_length
                    and self.ready_to_train):
                    break
#设置——准备好训练
                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')
#开始采样
                self._do_sampling(timestep=self._total_timestep)
                gt.stamp('sample')

                # print('epoch: {} | timestep: {} | total timesteps: {}'.format(self._epoch, self._timestep, self._epoch_length))
#如果准备好训练，开始训练，self._do_training_repeats(timestep=self._total_timestep)
                if self.ready_to_train:
                    self._do_training_repeats(timestep=self._total_timestep)
                gt.stamp('train')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')
#训练的路径，采样——得到最后的n个路径
            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))
            gt.stamp('training_paths')
#评估的路径，评估的路径里包含策略和评估环境        
                 
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment)
            gt.stamp('evaluation_paths')
 #训练的特征是由评估的rollouts定义的，评估的特征里有训练的路径和训练的环境
            training_metrics = self._evaluate_rollouts(
                training_paths, training_environment)
            gt.stamp('training_metrics')
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}
#还没有定义训练的时候用的算法，重要，我是不是该直接改个环境？？？还是该改评估的路径？还是应该改它的策略

            self._epoch_after_hook(training_paths)
            gt.stamp('epoch_after_hook')
#开始采样——诊断信息：策略的平均值、最小值、最大值、方差等
# """Return diagnostic information of the policy.
 #       Returns the mean, min, max, and standard deviation of means and
 #       covariances.
 #       """采样-诊断，迭代数、评估批、训练路径、评估路径
            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            time_diagnostics = gt.get_times().stamps.itrs

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'training/{key}', training_metrics[key])
                    for key in sorted(training_metrics.keys())
                ),
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                *(
                    (f'sampler/{key}', sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
            )))

            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                # TODO(hartikainen): Make this consistent such that there's no
                # need for the hasattr check.
                training_environment.render_rollouts(evaluation_paths)

            yield diagnostics
#训练的采样终止
        self.sampler.terminate()

        self._training_after_hook()

        yield {'done': True, **diagnostics}
#评估的路径，传入策略和评估环境。
    def _evaluation_paths(self, policy, evaluation_env):
        if self._eval_n_episodes < 1: return ()
#策略里面设置策略为deterministic，评估的时候传入deterministic，还有视频的保存
        with policy.set_deterministic(self._eval_deterministic):
            paths = rollouts(
                self._eval_n_episodes,
                evaluation_env,
                policy,
                self.sampler._max_path_length,
                render_mode=self._eval_render_mode)

        should_save_video = (
            self._video_save_frequency > 0
            and self._epoch % self._video_save_frequency == 0)

        if should_save_video:
            for i, path in enumerate(paths):
                video_frames = path.pop('images')
                video_file_name = f'evaluation_path_{self._epoch}_{i}.avi'
                video_file_path = os.path.join(
                    os.getcwd(), 'videos', video_file_name)
                save_video(video_frames, video_file_path)

        return paths
#评估的展开，rollout，为已经有的rollouts计算评估特征，里面有相关的路径和环境
#所有的return，episode的长度，诊断信息，环境信息
    def _evaluate_rollouts(self, paths, env):
        """Compute evaluation metrics for the given rollouts."""

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]

        diagnostics = OrderedDict((
            ('return-average', np.mean(total_returns)),
            ('return-min', np.min(total_returns)),
            ('return-max', np.max(total_returns)),
            ('return-std', np.std(total_returns)),
            ('episode-length-avg', np.mean(episode_lengths)),
            ('episode-length-min', np.min(episode_lengths)),
            ('episode-length-max', np.max(episode_lengths)),
            ('episode-length-std', np.std(episode_lengths)),
        ))

        env_infos = env.get_path_infos(paths)
        for key, value in env_infos.items():
            diagnostics[f'env_infos/{key}'] = value

        return diagnostics
#得到诊断信息
    @abc.abstractmethod
    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        raise NotImplementedError
#准备好训练
    @property
    def ready_to_train(self):
        return self.sampler.batch_ready()
#开始采样，sample
    def _do_sampling(self, timestep):
        self.sampler.sample()
#重复训练
    def _do_training_repeats(self, timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        if timestep % self._train_every_n_steps > 0: return
        trained_enough = (
            self._train_steps_this_epoch
            > self._max_train_repeat_per_timestep * self._timestep)
        if trained_enough: return

        for i in range(self._n_train_repeat):
            self._do_training(
                iteration=timestep,
                batch=self._training_batch())

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat
#开始训练
    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError
#初始化训练
    @abc.abstractmethod
    def _init_training(self):
        raise NotImplementedError
#保存tf
    @property
    def tf_saveables(self):
        return {}
#得到状态，状态里面有（epoch的长度，epoch，时间步，训练步数）
    def __getstate__(self):
        state = {
            '_epoch_length': self._epoch_length,
            '_epoch': (
                self._epoch + int(self._timestep >= self._epoch_length)),
            '_timestep': self._timestep % self._epoch_length,
            '_num_train_steps': self._num_train_steps,
        }

        return state
#定义状态，状态里面是字典
    def __setstate__(self, state):
        self.__dict__.update(state)
