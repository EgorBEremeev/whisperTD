class Ct2GenerationMixin:
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in
    [`Ct2PreTrainedModel`].

    The class exposes [`~generation.FlaxGenerationMixin.generate`], which can be used for:
            - *greedy decoding* by calling [`~generation.FlaxGenerationMixin._greedy_search`] if `num_beams=1` and
              `do_sample=False`
            - *multinomial sampling* by calling [`~generation.FlaxGenerationMixin._sample`] if `num_beams=1` and
              `do_sample=True`
            - *beam-search decoding* by calling [`~generation.FlaxGenerationMixin._beam_search`] if `num_beams>1` and
              `do_sample=False`

    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """