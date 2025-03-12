from collections import deque
import os
import io
import pickle
import socket
import torch
import shutil

class StorageManager:
    def __init__(self, name, load_session, load_episode, device, stage):
        if load_session and name not in load_session:
            print("ERROR: wrong combination of command and model! make sure command is: {}_agent".format(name))
            while True:
                pass
        self.machine_dir = os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_drl/model/' + str(socket.gethostname())
        if 'examples' in load_session:
            self.machine_dir = os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_drl/model/'
        self.name = name
        self.stage = load_session[-1] if load_session else stage
        self.session = load_session
        self.load_episode = load_episode
        self.session_dir = os.path.join(self.machine_dir, self.session)
        self.map_location = device

    def new_session_dir(self, stage):
        i = 0
        session_dir = os.path.join(self.machine_dir, f"{self.name}_{i}_stage_{stage}")
        while os.path.exists(session_dir):
            i += 1
            session_dir = os.path.join(self.machine_dir, f"{self.name}_{i}_stage_{stage}")
        self.session = f"{self.name}_{i}"
        print(f"Making new model dir: {session_dir}")
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir

    def delete_file(self, path):
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted file: {path}")

    # ------------------------------- SAVING -------------------------------

    def network_save_weights(self, network, model_dir, stage, episode):
        filepath = os.path.join(model_dir, f"{network.name}_stage{stage}_episode{episode}.pt")
        print(f"Saving {network.name} model for episode: {episode} at {filepath}")
        try:
            torch.save(network.state_dict(), filepath)
        except Exception as e:
            print(f"Failed to save network weights: {e}")

    def save_vae_weights(self, vae_models, model_dir, episode):
        for idx, vae in enumerate(vae_models, start=1):
            filepath = os.path.join(model_dir, f"vae_model{idx}_episode{episode}.pth")
            print(f"Saving VAE model {idx} for episode: {episode} at {filepath}")
            try:
                torch.save(vae.state_dict(), filepath)
            except Exception as e:
                print(f"Failed to save VAE model {idx}: {e}")

    def save_session(self, episode, networks, pickle_data, vae_models=None):
        print(f"Saving data for episode: {episode}, location: {self.session_dir}")
        for network in networks:
            self.network_save_weights(network, self.session_dir, self.stage, episode)
        if vae_models:
            self.save_vae_weights(vae_models, self.session_dir, episode)

        # Store graph data
        graph_path = os.path.join(self.session_dir, f"stage{self.stage}_episode{episode}.pkl")
        try:
            with open(graph_path, 'wb') as f:
                pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)
            print(f"Graph data saved at {graph_path}")
        except Exception as e:
            print(f"Failed to save graph data: {e}")

        # Delete previous iterations (except every 1000th episode)
        if episode % 1000 == 0:
            for i in range(episode, episode - 1000, -100):
                for network in networks:
                    path = os.path.join(self.session_dir, f"{network.name}_stage{self.stage}_episode{i}.pt")
                    self.delete_file(path)
                pkl_path = os.path.join(self.session_dir, f"stage{self.stage}_episode{i}.pkl")
                self.delete_file(pkl_path)

    def store_model(self, model):
        model_path = os.path.join(self.session_dir, f"stage{self.stage}_agent.pkl")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
            print(f"Model stored at {model_path}")
        except Exception as e:
            print(f"Failed to store model: {e}")

    # ------------------------------- LOADING -------------------------------

    def network_load_weights(self, network, model_dir, stage, episode):
        filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
        print(f"loading: {network.name} model from file: {filepath}")
        network.load_state_dict(torch.load(filepath, self.map_location))

    def load_vae_weights(self, vae_models, model_dir, episode):
        for idx, vae in enumerate(vae_models, start=1):
            filepath = os.path.join(model_dir, f"vae_model{idx}_episode{episode}.pth")
            print(f"loading VAE model {idx} from file: {filepath}")
            vae.load_state_dict(torch.load(filepath, self.map_location))

    def load_graphdata(self):
        with open(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.pkl'), 'rb') as f:
            return pickle.load(f)

    def load_replay_buffer(self, size, buffer_path):
        buffer_path = os.path.join(self.machine_dir, buffer_path)
        if (os.path.exists(buffer_path)):
            with open(buffer_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"buffer does not exist: {buffer_path}")
            return deque(maxlen=size)

    def load_model(self):
        model_path = os.path.join(self.session_dir, 'stage'+str(self.stage)+'_agent.pkl')
        try :
            with open(model_path, 'rb') as f:
                return CpuUnpickler(f, self.map_location).load()
        except FileNotFoundError:
            quit(f"The specified model: {model_path} was not found. Check whether you specified the correct stage {self.stage} and model name")

    def load_weights(self, networks, vae_models=None):
        for network in networks:
            self.network_load_weights(network, self.session_dir, self.stage, self.load_episode)
        if vae_models:
            self.load_vae_weights(vae_models, self.session_dir, self.load_episode)

class CpuUnpickler(pickle.Unpickler):
    def __init__(self, file, map_location):
        self.map_location = map_location
        super(CpuUnpickler, self).__init__(file)
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.map_location)
        else:
            return super().find_class(module, name)
        

class CpuUnpickler(pickle.Unpickler):
    def __init__(self, file, map_location):
        self.map_location = map_location
        super(CpuUnpickler, self).__init__(file)
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.map_location)
        else:
            return super().find_class(module, name)