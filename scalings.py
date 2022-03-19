##############################
# Class to handle scaling of intermediate layers during training
# Hypothesis: intermediate layers can arbitrarily scale and unscale their outputs, thwarting attempts to
#             regularize permutation sensitivity according to the distance between vectors.
##############################

class Scalings:
    """
    A structure that holds the configuration of our scaling strategies
    """
    def __init__(self, penalize_embedding, quantize, embedding_penalty=None):
        """
        :params: penalize_embedding -- add an L2 penalty for the graph representation with strength
                embedding_penalty
        :param: quantize -- Add standard normal noise to each coordinate of the embedding, independently,
                            at training time.
        """
        assert isinstance(quantize, bool)
        assert isinstance(penalize_embedding, bool)
        if penalize_embedding:
            assert embedding_penalty is not None
            # Convert penalty to float, could be string from command line args.
            embedding_penalty = float(embedding_penalty)
            assert embedding_penalty >= 0.

        self.penalize_embedding = penalize_embedding
        self.embedding_penalty = embedding_penalty
        self.quantize = quantize

    @property
    def scaling_active(self):
        return self.penalize_embedding or self.quantize

    def __str__(self):
        _str = "Scaling structure:\n\t"
        if self.scaling_active:
            for kk, vv in self.__dict__.items():
                _str += f"{kk}: {vv}\n\t"
        else:
            _str += "No scaling (e.g. no quantization)."

        return _str



