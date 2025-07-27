import numpy as np
from arsenal import colors
from arsenal.maths import log1mexp, logsumexp
from genlm.control.sampler.token import TokenSampler


class AWRS(TokenSampler):
    """Samples individual tokens through an adaptive weighted rejection sampling algorithm.

    This sampler is based on the algorithm described in [Fast Controlled Generation from Language Models w`ith Adaptive Weighted Rejection Sampling](https://arxiv.org/abs/2504.05410)

    Args:
        potential (genlm.control.Potential): The non-boolean potential.
        condition (genlm.control.Potential): The boolean condition. This potential must only output boolean values (0 or -inf in log-space).
        seed (int, optional): The seed for the random number generator. Defaults to None.
        prune_logws (bool, optional): Whether to prune the logws to only include the tokens in the intersection of the potential and condition vocabularies. Defaults to True.
        proper_weights (bool, optional): Whether to return properly weighted samples.
            If False, the sampler will only run one round of adaptive rejection sampling. Defaults to True.
    """

    def __init__(
        self, potential, condition, seed=None, prune_logws=True, proper_weights=True
    ):
        super().__init__(target=potential * condition)
        self.potential = potential
        self.condition = condition

        self.prune_logws = prune_logws
        self.proper_weights = proper_weights
        self.valid_idxs = np.array(
            [self.potential.lookup[t] for t in self.target.vocab_eos]
        )

        self.vocab_eos_set = set(self.target.vocab_eos)
        self.V = len(self.potential.vocab_eos)
        self.rng = np.random.default_rng(seed=seed)

    def _prune_logws(self, logws):
        # Prune the logws to only include the tokens in the
        # target vocabulary. (This zeros-out tokens which we know a priori
        # will be rejected.) Note: We need an additional correction term
        # to account for the fact that we're throwing away some probability mass.
        # This should be handled in `sample`.
        pruned = self.potential.alloc_logws()
        pruned[self.valid_idxs] = logws.weights[self.valid_idxs]
        logws.weights = pruned
        return logws

    async def _accept(self, context, token, verbosity=0):
        if self.prune_logws or token in self.vocab_eos_set:
            if token is self.target.eos:
                logscore = await self.condition.complete(context)
            else:
                logscore = await self.condition.prefix(context + [token])
            assert logscore in {-np.inf, 0}, "`condition` must be Boolean"
        else:
            logscore = -np.inf

        do_accept = logscore == 0

        if verbosity > 0:
            if do_accept:
                print(colors.green % f". {repr(token)}")
            else:
                print(colors.red % ".", end="")

        return do_accept

    async def sample(self, context, verbosity=0):
        """Sample a token and weight that are properly weighted with respect to the target potential's `logw_next` method via adaptive weighted rejection sampling.

        The returned weight corresponds to the log normalizing constant of $\\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})$.

        Returns:
            (token, weight, np.nan): A tuple containing the sampled token, weight, and a dummy value for the log-probability of the sampled token.
        """
        logws = await self.potential.logw_next(context)
        if self.prune_logws:
            logws = self._prune_logws(logws)

        logZ = logsumexp(logws.weights)
        logps = logws.weights - logZ
        toks = logws.decode

        tok, nrej, logp0 = None, 0, []
        for _ in range(2):
            keys = logps - np.log(-np.log(self.rng.random((self.V,))))
            order = np.argsort(-keys)
            for rank in range(logps.size):
                item = order[rank]
                if keys[item] == -np.inf:
                    break
                if await self._accept(context, toks[item], verbosity):
                    if tok is None:
                        tok = toks[item]
                    break
                else:
                    nrej += 1
                    if tok is None:
                        logp0.append(logps[item])
                    logps[item] = -np.inf

            if not self.proper_weights:
                if tok is None:
                    return self.target.eos, float("-inf"), np.nan
                return tok, 0, np.nan

        if tok is None:  # No token was accepted, return EOS and kill the particle.
            return self.target.eos, float("-inf"), np.nan

        if not logp0:  # Success on first try.
            logw = logZ - np.log(nrej + 1)
        else:
            logw = logZ + log1mexp(logsumexp(logp0)) - np.log(nrej + 1)

        return tok, logw, np.nan
