import gymnasium as gym
from env import BlockBlastEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
import os

def mask_fn(env: gym.Env):
    return env.valid_action_mask()

# === NOUL INTERCEPTOR PENTRU TENSORBOARD ===
class TensorboardStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Se uită în info dict la fiecare pas
        for info in self.locals.get("infos", []):
            if "game/etap_max" in info:
                # Logăm datele! Vor apărea într-o categorie nouă numită "game_stats"
                self.logger.record("game_stats/Etape_Supravietuite", info["game/etap_max"])
                self.logger.record("game_stats/Linii_Distruse", info["game/linii_distruse"])
                self.logger.record("game_stats/Blocuri_Puse", info["game/blocuri_puse"])
        return True

if __name__ == "__main__":
    print("Initializez mediul Block Blast...")
    raw_env = BlockBlastEnv()
    env = ActionMasker(raw_env, mask_fn)

    model_path = "block_blast_ai_v1"
    log_dir = "./tensorboard_logs/"

    if os.path.exists(model_path + ".zip"):
        print("✅ Continuăm antrenamentul...")
        model = MaskablePPO.load(model_path, env=env, tensorboard_log=log_dir)
    else:
        print("🧠 Inițializăm un creier NOU...")
        model = MaskablePPO(
            "MultiInputPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0003, 
            gamma=0.99,
            tensorboard_log=log_dir
        )

    print("Începem antrenamentul...")
    
    # AM ADĂUGAT CALLBACK-UL AICI
    stats_callback = TensorboardStatsCallback()
    model.learn(total_timesteps=1_000_000, tb_log_name="PPO_BlockBlast", reset_num_timesteps=False, callback=stats_callback)

    print("Antrenament complet! Salvăm modelul...")
    model.save(model_path)