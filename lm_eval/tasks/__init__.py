from pprint import pprint
from typing import List, Union

import sacrebleu
import lm_eval.base


from . import arc

from . import assin
from . import faquad
from . import big_bench
from . import tweetsentbr
from . import enem
from . import massive
from . import mkqa
from . import agnewspt
from . import imdbpt
from . import sst2_pt
from . import boolq_pt
from . import wsc285_pt
from . import bluex
from . import poscomp
from . import hatebr
from . import storycloze_pt
from . import pt_hate_speech

########################################
# Translation tasks
########################################

# 6 total
gpt3_translation_benchmarks = {
    "wmt14": ['en-fr', 'fr-en'],  # French
    "wmt16": ['en-ro', 'ro-en', 'de-en', 'en-de'],  # German, Romanian
}


# 28 total
selected_translation_benchmarks = {
    **gpt3_translation_benchmarks,
    "wmt20": sacrebleu.get_langpairs_for_testset("wmt20"),
    "iwslt17": ['en-ar', 'ar-en']  # Arabic
}

# 319 total
all_translation_benchmarks = {
    ts: sacrebleu.get_langpairs_for_testset(ts)
    for ts in sacrebleu.get_available_testsets()
}


########################################
# All tasks
########################################


TASK_REGISTRY = {
    
    "broverbs_proverb_to_history_greedy": big_bench.BroverbsProverbToHistoryTaskGreedy,
    "broverbs_history_to_proverb_greedy": big_bench.BroverbsHistoryToProverbTaskGreedy,
    "bigbench_pt_analogical_similarity_greedy": big_bench.AnalogicalSimilarityTaskGreedy,
    "bigbench_pt_code_line_description_greedy": big_bench.CodeLineDescriptionTaskGreedy,
    "bigbench_pt_contextual_parametric_knowledge_conflicts_greedy": big_bench.ContextualParametricKnowledgeConflictsTaskGreedy,
    "bigbench_pt_dark_humor_detection_greedy": big_bench.DarkHumorDetectionTaskGreedy,
    "bigbench_pt_empirical_judgments_greedy": big_bench.EmpiricalJudgmentsTaskGreedy,
    "bigbench_pt_formal_fallacies_syllogisms_negation_greedy": big_bench.FormalFallaciesSyllogismsNegationTaskGreedy,
    "bigbench_pt_general_knowledge_greedy": big_bench.GeneralKnowledgeTaskGreedy,
    "bigbench_pt_mathematical_induction_greedy": big_bench.MathematicalInductionTaskGreedy,
    "bigbench_pt_simple_ethical_questions_greedy": big_bench.SimpleEthicalQuestionsTaskGreedy,
    "bigbench_pt_strategyqa_greedy": big_bench.StrategyQATaskGreedy,
    "bigbench_pt_vitaminc_fact_verification_greedy": big_bench.VitamincFactVerificationTaskGreedy,
    "bigbench_pt_causal_judgment_greedy": big_bench.CausalJudgmentTaskGreedy,
    "bigbench_pt_cause_and_effect_two_sentences_greedy": big_bench.CauseAndEffectTwoSentencesTaskGreedy,
    "bigbench_pt_bbq_greedy": big_bench.BbqTaskGreedy,
    "bigbench_pt_social_iqa_greedy": big_bench.SocialIQATaskGreedy,
    
    # bigbench en
    "bigbench_en_analogical_similarity_greedy": big_bench.EnglishOriginalAnalogicalSimilarityTaskGreedy,
    "bigbench_en_code_line_description_greedy": big_bench.EnglishOriginalCodeLineDescriptionTaskGreedy,
    "bigbench_en_contextual_parametric_knowledge_conflicts_greedy": big_bench.EnglishOriginalContextualParametricKnowledgeConflictsTaskGreedy,
    "bigbench_en_dark_humor_detection_greedy": big_bench.EnglishOriginalDarkHumorDetectionTaskGreedy,
    "bigbench_en_empirical_judgments_greedy": big_bench.EnglishOriginalEmpiricalJudgmentsTaskGreedy,
    "bigbench_en_formal_fallacies_syllogisms_negation_greedy": big_bench.EnglishOriginalFormalFallaciesSyllogismsNegationTaskGreedy,
    "bigbench_en_general_knowledge_greedy": big_bench.EnglishOriginalGeneralKnowledgeTaskGreedy,
    "bigbench_en_mathematical_induction_greedy": big_bench.EnglishOriginalMathematicalInductionTaskGreedy,
    "bigbench_en_simple_ethical_questions_greedy": big_bench.EnglishOriginalSimpleEthicalQuestionsTaskGreedy,
    "bigbench_en_strategyqa_greedy": big_bench.EnglishOriginalStrategyQATaskGreedy,
    "bigbench_en_vitaminc_fact_verification_greedy": big_bench.EnglishOriginalVitamincFactVerificationTaskGreedy,
    "bigbench_en_cause_and_effect_two_sentences_greedy": big_bench.EnglishOriginalCauseAndEffectTwoSentencesTaskGreedy,
    "bigbench_en_bbq_greedy": big_bench.EnglishOriginalBbqTaskGreedy,
    "bigbench_en_social_iqa_greedy": big_bench.EnglishOriginalSocialIQATaskGreedy,
    "bigbench_en_causal_judgment_greedy": big_bench.EnglishOriginalCausalJudgmentTaskGreedy,
    
    # not actually from bigbench, just using the same format
    "ethics_commonsense_test_hard_greedy": big_bench.EthicsCommonSenseHardTaskGreedy,
    "inferbr_greedy": big_bench.InferBRTaskGreedy,
    "repro_greedy": big_bench.ReproTaskGreedy,
    "mina_br_greedy": big_bench.MinaBRTaskGreedy,
    "math_mc_greedy": big_bench.MathMCTaskGreedy,
    "gsm8k_mc_greedy": big_bench.GSM8KMCTaskGreedy,
    "agieval_sat_math_greedy": big_bench.SATMathTaskGreedy,
    "balanced_copa_greedy": big_bench.BalancedCopaTaskGreedy,
    "logiqa_greedy": big_bench.LogiQATaskGreedy,
    
    "assin_rte": assin.ASSIN_RTE,
    "assin_rte_greedy": assin.ASSIN_RTE_GREEDY,
    "assin_sts": assin.ASSIN_STS,
    "assin_sts_greedy": assin.ASSIN_STS_GREEDY,

    "faquad": faquad.FAQuAD,

    "tweetsentbr": tweetsentbr.TweetSentBR,
    "tweetsentbr_greedy": tweetsentbr.TweetSentBR_GREEDY,

    "pt_hate_speech": pt_hate_speech.HateSpeechPT_Binary,
    "pt_hate_speech_greedy": pt_hate_speech.HateSpeechPT_Greedy,

    "hatebr_binary": hatebr.HateBR_Binary,
    "hatebr_multi": hatebr.HateBR_multiclass,
    "hatebr_binary_greedy": hatebr.HateBR_Binary_Greedy,
    "hatebr_multi_greedy": hatebr.HateBR_multiclass_Greedy,

    # StoryCloze pt
    "storycloze_pt": storycloze_pt.StoryClozePT,
    "storycloze_pt_greedy": storycloze_pt.StoryClozePT_Greedy,

    "bluex": bluex.BLUEX,
    "bluex_greedy": bluex.BLUEX_GREEDY,
    "bluex_recent": bluex.BLUEX_RECENT,
    "bluex_launch_version_greedy": bluex.BLUEX_LAUNCH_VERSION_GREEDY,

    "poscomp": poscomp.POSCOMP,
    "poscomp_greedy": poscomp.POSCOMP_GREEDY,
    "poscomp_recent": poscomp.POSCOMP_RECENT,
    "poscomp_recent_greedy": poscomp.POSCOMP_RECENT_GREEDY,

    "enem": enem.ENEM,
    "enem_2022": enem.ENEM_2022,
    "enem_greedy": enem.ENEM_GREEDY,
    "enem_2022_greedy": enem.ENEM_2022_GREEDY,

    "massive": massive.MASSIVE,
    "massive_greedy": massive.MASSIVE_GREEDY,

    "mkqa": mkqa.MKQA,
    "mkqa_greedy": mkqa.MKQA_GREEDY,

    "agnews_pt": agnewspt.AGNewsPT,
    "agnews_pt_greedy": agnewspt.AGNewsPT_GREEDY,

    "imdb_pt": imdbpt.IMDBPT,
    "imdb_pt_greedy": imdbpt.IMDBPT_GREEDY,

    "sst2_pt": sst2_pt.SST2PT,
    "sst2_pt_greedy": sst2_pt.SST2PT_GREEDY,

    "boolq_pt": boolq_pt.BOOLQPT,
    "boolq_pt_greedy": boolq_pt.BOOLQPT_GREEDY,

    "wsc285_pt": wsc285_pt.WinogradSchemaChallenge285,
    "wsc285_pt_greedy": wsc285_pt.WinogradSchemaChallenge285_GREEDY,

    "arc_challenge_greedy": arc.ARC_CHALLENGE_greedy,
    "arc_challenge_greedy_pt": arc.ARC_CHALLENGE_greedy_PT,
    "arc_easy_greedy": arc.ARC_EASY_greedy,
    "arc_easy_greedy_pt": arc.ARC_EASY_greedy_PT,
    
    
    
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as e:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return task_object.EVAL_HARNESS_NAME if hasattr(task_object, "EVAL_HARNESS_NAME") else type(task_object).__name__


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
