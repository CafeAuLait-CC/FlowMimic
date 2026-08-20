"""T2M motion and text feature extractors used by common evaluation metrics.

Source reference:
- motion-latent-diffusion (preferred source): https://github.com/ChenFengYe/motion-latent-diffusion
  Reused/adapted from: mld/models/architectures/t2m_motionenc.py
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=True)
        )

    def forward(self, inputs, lengths):
        num_samples = inputs.shape[0]
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        packed = pack_padded_sequence(
            input_embs, lengths.data.tolist(), batch_first=True, enforce_sorted=True
        )
        _, gru_last = self.gru(packed, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


class TextEncoderBiGRUCo(nn.Module):
    def __init__(
        self,
        word_size=300,
        pos_size=15,
        hidden_size=512,
        output_size=512,
    ):
        super().__init__()
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, hidden_size), requires_grad=True)
        )

    def forward(self, word_embeddings, pos_one_hot, lengths):
        num_samples = word_embeddings.shape[0]
        inputs = word_embeddings + self.pos_emb(pos_one_hot)
        inputs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        packed = pack_padded_sequence(
            inputs,
            lengths.data.tolist(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, last = self.gru(packed, hidden)
        last = torch.cat([last[0], last[1]], dim=-1)
        return self.output_net(last)


POS_ENUMERATOR = {
    "VERB": 0,
    "NOUN": 1,
    "DET": 2,
    "ADP": 3,
    "NUM": 4,
    "AUX": 5,
    "PRON": 6,
    "ADJ": 7,
    "ADV": 8,
    "Loc_VIP": 9,
    "Body_VIP": 10,
    "Obj_VIP": 11,
    "Act_VIP": 12,
    "Desc_VIP": 13,
    "OTHER": 14,
}

VIP_WORDS = {
    "Loc_VIP": {
        "left", "right", "clockwise", "counterclockwise", "anticlockwise",
        "forward", "back", "backward", "up", "down", "straight", "curve",
    },
    "Body_VIP": {
        "arm", "chin", "foot", "feet", "face", "hand", "mouth", "leg",
        "waist", "eye", "knee", "shoulder", "thigh",
    },
    "Obj_VIP": {
        "stair", "dumbbell", "chair", "window", "floor", "car", "ball",
        "handrail", "baseball", "basketball",
    },
    "Act_VIP": {
        "walk", "run", "swing", "pick", "bring", "kick", "put", "squat",
        "throw", "hop", "dance", "jump", "turn", "stumble", "stop", "sit",
        "lift", "lower", "raise", "wash", "stand", "kneel", "stroll", "rub",
        "bend", "balance", "flap", "jog", "shuffle", "lean", "rotate", "spin",
        "spread", "climb",
    },
    "Desc_VIP": {
        "slowly", "carefully", "fast", "careful", "slow", "quickly", "happy",
        "angry", "sad", "happily", "angrily", "sadly",
    },
}


class T2MWordVectorizer:
    """HumanML3D GloVe/POS vectorizer used by the paired AIST evaluator."""

    def __init__(self, root, prefix="our_vab"):
        root = Path(root)
        vectors = np.load(root / f"{prefix}_data.npy")
        with (root / f"{prefix}_words.pkl").open("rb") as handle:
            words = pickle.load(handle)
        with (root / f"{prefix}_idx.pkl").open("rb") as handle:
            word_to_index = pickle.load(handle)
        self.word_to_vector = {
            word: vectors[word_to_index[word]].astype(np.float32, copy=False)
            for word in words
        }

    @staticmethod
    def _pos_vector(pos):
        result = np.zeros((len(POS_ENUMERATOR),), dtype=np.float32)
        result[POS_ENUMERATOR.get(pos, POS_ENUMERATOR["OTHER"])] = 1.0
        return result

    def __getitem__(self, token):
        try:
            word, pos = token.rsplit("/", 1)
        except ValueError:
            word, pos = token, "OTHER"
        if word not in self.word_to_vector:
            return self.word_to_vector["unk"], self._pos_vector("OTHER")
        vip_pos = next(
            (name for name, words in VIP_WORDS.items() if word in words),
            None,
        )
        return self.word_to_vector[word], self._pos_vector(vip_pos or pos)

    def batch(self, token_sequences, max_text_len=20):
        word_batches = []
        pos_batches = []
        lengths = []
        for sequence in token_sequences:
            tokens = sequence.split() if isinstance(sequence, str) else list(sequence)
            if len(tokens) < max_text_len:
                tokens = ["sos/OTHER", *tokens, "eos/OTHER"]
                length = len(tokens)
                tokens.extend(["unk/OTHER"] * (max_text_len + 2 - length))
            else:
                tokens = ["sos/OTHER", *tokens[:max_text_len], "eos/OTHER"]
                length = len(tokens)
            words, positions = zip(*(self[token] for token in tokens))
            word_batches.append(np.stack(words))
            pos_batches.append(np.stack(positions))
            lengths.append(length)
        return (
            np.stack(word_batches).astype(np.float32),
            np.stack(pos_batches).astype(np.float32),
            np.asarray(lengths, dtype=np.int64),
        )


class T2MMotionFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_size=263,
        movement_hidden_size=512,
        movement_latent_size=512,
        motion_hidden_size=1024,
        motion_latent_size=512,
    ):
        super().__init__()
        self.movement_encoder = MovementConvEncoder(
            input_size=input_size - 4,
            hidden_size=movement_hidden_size,
            output_size=movement_latent_size,
        )
        self.motion_encoder = MotionEncoderBiGRUCo(
            input_size=movement_latent_size,
            hidden_size=motion_hidden_size,
            output_size=motion_latent_size,
        )

    def load_pretrained(self, ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        if "movement_encoder" in state and "motion_encoder" in state:
            move_state = state["movement_encoder"]
            motion_state = state["motion_encoder"]
        elif "state_dict" in state:
            sd = state["state_dict"]
            move_state = {
                k.replace("movement_encoder.", "", 1): v
                for k, v in sd.items()
                if k.startswith("movement_encoder.")
            }
            motion_state = {
                k.replace("motion_encoder.", "", 1): v
                for k, v in sd.items()
                if k.startswith("motion_encoder.")
            }
            if not move_state or not motion_state:
                raise KeyError(
                    "Checkpoint state_dict does not contain movement_encoder/motion_encoder keys"
                )
        else:
            raise KeyError(
                "Unsupported T2M checkpoint format: expected movement_encoder/motion_encoder"
            )
        self.movement_encoder.load_state_dict(move_state)
        self.motion_encoder.load_state_dict(motion_state)

    @torch.no_grad()
    def encode(self, motion, motion_length):
        # motion: [B, T, 263], motion_length: [B]
        motion = motion.float()
        motion_length = motion_length.long()
        sort_idx = np.argsort(motion_length.cpu().numpy())[::-1].copy()
        rank_idx = np.empty_like(sort_idx)
        rank_idx[sort_idx] = np.arange(len(sort_idx))
        sort_idx_t = torch.from_numpy(sort_idx).to(motion.device, dtype=torch.long)
        rank_idx_t = torch.from_numpy(rank_idx).to(motion.device, dtype=torch.long)

        motion_sorted = motion[sort_idx_t]
        length_sorted = motion_length[sort_idx_t]
        movement = self.movement_encoder(motion_sorted[..., :-4]).detach()
        movement_lens = torch.clamp(length_sorted // 4, min=1)
        emb_sorted = self.motion_encoder(movement, movement_lens)
        return emb_sorted[rank_idx_t]


class T2MTextFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoderBiGRUCo()

    def load_pretrained(self, ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "text_encoder" not in state:
            raise KeyError("T2M checkpoint does not contain text_encoder")
        self.text_encoder.load_state_dict(state["text_encoder"])

    @torch.no_grad()
    def encode(self, word_embeddings, pos_one_hot, lengths):
        return self.text_encoder(
            word_embeddings.float(), pos_one_hot.float(), lengths.long()
        )
