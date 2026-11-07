import json,re
from pathlib import Path
import torch
import torch.nn as nn
CHAR_EMBEDDING_DIM=50
CHAR_HIDDEN_DIM=50
EMBEDDING_DIM=128
WORD_EMBEDDING_DIM=128
HIDDEN_DIM=128
WORD_HIDDEN_DIM=128
DROPOUT=0.2
PAD_ID=0
UNK_ID=1
CHAR_PAD_ID=0
CHAR_UNK_ID=1

class CRF(
    nn.Module
):

    def __init__(
        self,
        num_tags
    ):

        super().__init__()


        self.num_tags = num_tags


        self.transitions = nn.Parameter(
            torch.randn(
                num_tags,
                num_tags
            ) * 0.1
        )


        self.start_transitions = nn.Parameter(
            torch.zeros(
                num_tags
            )
        )


        self.end_transitions = nn.Parameter(
            torch.zeros(
                num_tags
            )
        )


    # ========================================================
    # GOLD SCORE
    # ========================================================

    def score_sentence(
        self,
        emissions,
        tags,
        mask
    ):

        batch_size, seq_len, _ = (
            emissions.size()
        )


        device = emissions.device


        batch_indices = torch.arange(
            batch_size,
            device=device
        )


        first_tags = tags[:, 0]


        scores = (
            self.start_transitions[
                first_tags
            ]
            +
            emissions[
                batch_indices,
                0,
                first_tags
            ]
        )


        for t in range(
            1,
            seq_len
        ):

            valid = mask[:, t]


            if not valid.any():

                continue


            valid_indices = torch.nonzero(
                valid,
                as_tuple=False
            ).squeeze(1)


            previous_tags = tags[
                valid_indices,
                t - 1
            ]


            current_tags = tags[
                valid_indices,
                t
            ]


            scores[
                valid_indices
            ] += (

                self.transitions[
                    previous_tags,
                    current_tags
                ]

                +

                emissions[
                    valid_indices,
                    t,
                    current_tags
                ]
            )


        lengths = mask.sum(
            dim=1
        )


        last_indices = lengths - 1


        last_tags = tags[
            batch_indices,
            last_indices
        ]


        scores += self.end_transitions[
            last_tags
        ]


        return scores


    # ========================================================
    # PARTITION FUNCTION
    # ========================================================

    def compute_log_partition(
        self,
        emissions,
        mask
    ):

        batch_size, seq_len, num_tags = (
            emissions.size()
        )


        alpha = (
            self.start_transitions
            +
            emissions[:, 0]
        )


        for t in range(
            1,
            seq_len
        ):

            next_alpha = torch.logsumexp(
                alpha.unsqueeze(2)
                +
                self.transitions.unsqueeze(0)
                +
                emissions[
                    :,
                    t
                ].unsqueeze(1),
                dim=1
            )


            valid = mask[:, t]


            alpha = torch.where(
                valid.unsqueeze(1),
                next_alpha,
                alpha
            )


        alpha += self.end_transitions


        return torch.logsumexp(
            alpha,
            dim=1
        )


    # ========================================================
    # NEGATIVE LOG LIKELIHOOD
    # ========================================================

    def neg_log_likelihood(
        self,
        emissions,
        tags,
        mask
    ):

        gold_score = (
            self.score_sentence(
                emissions,
                tags,
                mask
            )
        )


        log_partition = (
            self.compute_log_partition(
                emissions,
                mask
            )
        )


        return (
            log_partition
            -
            gold_score
        ).mean()


    # ========================================================
    # VITERBI DECODE
    # ========================================================

    def decode(
        self,
        emissions,
        mask
    ):

        batch_size, seq_len, num_tags = (
            emissions.size()
        )


        predictions = []


        for batch_index in range(
            batch_size
        ):

            length = int(
                mask[
                    batch_index
                ].sum().item()
            )


            current = (
                self.start_transitions
                +
                emissions[
                    batch_index,
                    0
                ]
            )


            backpointers = []


            for t in range(
                1,
                length
            ):

                scores = (
                    current.unsqueeze(1)
                    +
                    self.transitions
                    +
                    emissions[
                        batch_index,
                        t
                    ].unsqueeze(0)
                )


                best_scores, best_previous = (
                    scores.max(
                        dim=0
                    )
                )


                current = best_scores


                backpointers.append(
                    best_previous
                )


            current += (
                self.end_transitions
            )


            best_last = int(
                current.argmax().item()
            )


            path = [
                best_last
            ]


            for backpointer in reversed(
                backpointers
            ):

                best_last = int(
                    backpointer[
                        best_last
                    ].item()
                )


                path.append(
                    best_last
                )


            path.reverse()


            predictions.append(
                path
            )


        return predictions


# ============================================================

class CharacterBiLSTMCRF(
    nn.Module
):

    def __init__(
        self,
        word_vocab_size,
        character_vocab_size,
        num_labels
    ):

        super().__init__()


        # ----------------------------------------------------
        # Character encoder
        # ----------------------------------------------------

        self.character_embedding = nn.Embedding(
            character_vocab_size,
            CHAR_EMBEDDING_DIM,
            padding_idx=CHAR_PAD_ID
        )


        self.character_lstm = nn.LSTM(
            input_size=CHAR_EMBEDDING_DIM,
            hidden_size=CHAR_HIDDEN_DIM,
            batch_first=True,
            bidirectional=True
        )


        # ----------------------------------------------------
        # Word encoder
        # ----------------------------------------------------

        self.word_embedding = nn.Embedding(
            word_vocab_size,
            EMBEDDING_DIM,
            padding_idx=PAD_ID
        )


        self.word_lstm = nn.LSTM(
            input_size=(
                EMBEDDING_DIM
                +
                (
                    2 * CHAR_HIDDEN_DIM
                )
            ),
            hidden_size=HIDDEN_DIM,
            batch_first=True,
            bidirectional=True
        )


        self.dropout = nn.Dropout(
            DROPOUT
        )


        # ----------------------------------------------------
        # Emission layer
        # ----------------------------------------------------

        self.emission = nn.Linear(
            2 * HIDDEN_DIM,
            num_labels
        )


        # ----------------------------------------------------
        # CRF
        # ----------------------------------------------------

        self.crf = CRF(
            num_labels
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        word_ids,
        character_ids
    ):

        batch_size, seq_len, max_chars = (
            character_ids.size()
        )


        # ----------------------------------------------------
        # Character encoding
        # ----------------------------------------------------

        char_embeddings = (
            self.character_embedding(
                character_ids
            )
        )


        char_embeddings = (
            char_embeddings.reshape(
                batch_size * seq_len,
                max_chars,
                CHAR_EMBEDDING_DIM
            )
        )


        _, (
            hidden,
            _
        ) = self.character_lstm(
            char_embeddings
        )


        forward_hidden = hidden[-2]

        backward_hidden = hidden[-1]


        character_representation = torch.cat(
            [
                forward_hidden,
                backward_hidden
            ],
            dim=1
        )


        character_representation = (
            character_representation.reshape(
                batch_size,
                seq_len,
                2 * CHAR_HIDDEN_DIM
            )
        )


        # ----------------------------------------------------
        # Word embedding
        # ----------------------------------------------------

        word_representation = (
            self.word_embedding(
                word_ids
            )
        )


        # ----------------------------------------------------
        # Combine word + character
        # ----------------------------------------------------

        combined = torch.cat(
            [
                word_representation,
                character_representation
            ],
            dim=2
        )


        # ----------------------------------------------------
        # Word-level BiLSTM
        # ----------------------------------------------------

        contextualized, _ = (
            self.word_lstm(
                combined
            )
        )


        contextualized = self.dropout(
            contextualized
        )


        # ----------------------------------------------------
        # Emissions
        # ----------------------------------------------------

        emissions = self.emission(
            contextualized
        )


        return emissions

class CharacterBiLSTMCRF(
    nn.Module
):

    def __init__(
        self,
        word_vocab_size,
        character_vocab_size,
        num_labels
    ):

        super().__init__()


        # ----------------------------------------------------
        # CHARACTER EMBEDDING
        # ----------------------------------------------------

        self.character_embedding = nn.Embedding(
            character_vocab_size,
            CHAR_EMBEDDING_DIM,
            padding_idx=CHAR_PAD_ID
        )


        # ----------------------------------------------------
        # CHARACTER BILSTM
        # ----------------------------------------------------

        self.character_lstm = nn.LSTM(
            input_size=CHAR_EMBEDDING_DIM,
            hidden_size=CHAR_HIDDEN_DIM,
            batch_first=True,
            bidirectional=True
        )


        # ----------------------------------------------------
        # WORD EMBEDDING
        # ----------------------------------------------------

        self.word_embedding = nn.Embedding(
            word_vocab_size,
            WORD_EMBEDDING_DIM,
            padding_idx=PAD_ID
        )


        # ----------------------------------------------------
        # WORD BILSTM
        # ----------------------------------------------------

        self.word_lstm = nn.LSTM(
            input_size=(
                WORD_EMBEDDING_DIM
                +
                2 * CHAR_HIDDEN_DIM
            ),
            hidden_size=WORD_HIDDEN_DIM,
            batch_first=True,
            bidirectional=True
        )


        self.dropout = nn.Dropout(
            DROPOUT
        )


        # ----------------------------------------------------
        # EMISSION
        # ----------------------------------------------------

        self.emission = nn.Linear(
            2 * WORD_HIDDEN_DIM,
            num_labels
        )


        # ----------------------------------------------------
        # CRF
        # ----------------------------------------------------

        self.crf = CRF(
            num_labels
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        word_ids,
        character_ids
    ):

        batch_size, seq_len, max_chars = (
            character_ids.size()
        )


        # ----------------------------------------------------
        # CHARACTER ENCODING
        # ----------------------------------------------------

        char_embeddings = (
            self.character_embedding(
                character_ids
            )
        )


        char_embeddings = (
            char_embeddings.reshape(
                batch_size * seq_len,
                max_chars,
                CHAR_EMBEDDING_DIM
            )
        )


        _, (
            hidden,
            _
        ) = self.character_lstm(
            char_embeddings
        )


        forward_hidden = hidden[-2]

        backward_hidden = hidden[-1]


        character_representation = torch.cat(
            [
                forward_hidden,
                backward_hidden
            ],
            dim=1
        )


        character_representation = (
            character_representation.reshape(
                batch_size,
                seq_len,
                2 * CHAR_HIDDEN_DIM
            )
        )


        # ----------------------------------------------------
        # WORD ENCODING
        # ----------------------------------------------------

        word_representation = (
            self.word_embedding(
                word_ids
            )
        )


        # ----------------------------------------------------
        # CONCATENATION
        # ----------------------------------------------------

        combined = torch.cat(
            [
                word_representation,
                character_representation
            ],
            dim=2
        )


        # ----------------------------------------------------
        # WORD BILSTM
        # ----------------------------------------------------

        contextualized, _ = (
            self.word_lstm(
                combined
            )
        )


        contextualized = self.dropout(
            contextualized
        )


        # ----------------------------------------------------
        # EMISSION
        # ----------------------------------------------------

        emissions = self.emission(
            contextualized
        )


        return emissions

class Predictor:
    def __init__(self, root):
        self.root=Path(root)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos,self.pw,self.pc,self.pl=self.load("pos")
        self.ner,self.nw,self.nc,self.nl=self.load("ner")
        self.pm=json.loads((self.root/"results/pos_char_bilstm_crf_results.json").read_text(encoding="utf-8"))["test"]
        self.nm=json.loads((self.root/"results/ner_char_bilstm_crf_results.json").read_text(encoding="utf-8"))["test"]
    def load(self,task):
        ck=torch.load(self.root/f"models/{task}_char_bilstm_crf.pt",map_location=self.device,weights_only=False)
        w,c,l=ck["word_to_id"],ck["char_to_id"],ck["label_to_id"]
        m=CharacterBiLSTMCRF(len(w),len(c),len(l))
        m.load_state_dict(ck["model_state_dict"]); m.to(self.device).eval()
        return m,w,c,l
    def analyze(self,text):
        tokens=re.findall(r"[^\\W_]+(?:['’][^\\W_]+)*|[^\\w\\s]",text,flags=re.UNICODE)
        if not tokens: raise ValueError("Enter a Ronglat word or sentence.")
        def run(m,wv,cv,lv):
            wi=[wv.get(t,UNK_ID) for t in tokens]
            ci=[[cv.get(ch,CHAR_UNK_ID) for ch in t] or [CHAR_PAD_ID] for t in tokens]
            mx=max(map(len,ci)); ci=[x+[CHAR_PAD_ID]*(mx-len(x)) for x in ci]
            words=torch.tensor([wi],dtype=torch.long,device=self.device)
            chars=torch.tensor([ci],dtype=torch.long,device=self.device)
            mask=torch.ones((1,len(tokens)),dtype=torch.bool,device=self.device)
            with torch.no_grad(): emissions=m(words,chars); ids=m.crf.decode(emissions,mask)[0]
            inv={v:k for k,v in lv.items()}
            return [inv[i] for i in ids]
        p=run(self.pos,self.pw,self.pc,self.pl); n=run(self.ner,self.nw,self.nc,self.nl)
        return [{"token":t,"pos":a,"ner":b,"known_word":t in self.pw or t in self.nw} for t,a,b in zip(tokens,p,n)]
