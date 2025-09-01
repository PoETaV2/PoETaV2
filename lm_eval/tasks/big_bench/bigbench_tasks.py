from .bigbench_task_base import BigBenchTaskBaseGreedy


class BroverbsProverbToHistoryTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "BRoverbs"
    LOCAL_PATH = "BRoverbs/BRoverbs_proverb_to_history.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/BRoverbs_proverb_to_history/task.json"
    task_type = "target_scores_to_alternatives"

    example_input_prefix = "Provérbio:"


class BroverbsHistoryToProverbTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "BRoverbs"
    LOCAL_PATH = "BRoverbs/BRoverbs_history_to_proverb.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/BRoverbs_history_to_proverb/task.json"
    task_type = "target_scores_to_alternatives"

    example_input_prefix = "História:"


# analogical_similarity
class AnalogicalSimilarityTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "analogical_similarity"
    LOCAL_PATH = "bigbench_pt/analogical_similarity/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/analogical_similarity/task.json"
    task_type = "target_scores_to_alternatives"

    class_mapping = {
        "literal similarity": "Similaridade Literal",
        "an analogy.": "Analogia",
        "a cross mapping.": "Mapeamento Cruzado",
        "surface similarity.": "Similaridade de Superfície",
        "a false analogy.": "Analogia Falsa",
        "only objects similarity.": "Similaridade de Objetos",
        "no similarity.": "Nenhuma Similaridade",
    }


# code_line_description
class CodeLineDescriptionTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "code_line_description"
    LOCAL_PATH = "bigbench_pt/code_line_description/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/code_line_description/task.json"
    task_type = "target_scores_to_alternatives"


# contextual_parametric_knowledge_conflicts
class ContextualParametricKnowledgeConflictsTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "contextual_parametric_knowledge_conflicts"
    LOCAL_PATH = "bigbench_pt/contextual_parametric_knowledge_conflicts/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/contextual_parametric_knowledge_conflicts/task.json"
    task_type = "target_scores_to_alternatives"


# dark_humor_detection
class DarkHumorDetectionTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "dark_humor_detection"
    LOCAL_PATH = "bigbench_pt/dark_humor_detection/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/dark_humor_detection/task.json"
    task_type = "target_scores_to_alternatives"

    class_mapping = {"joke": "Piada", "not a joke": "Não é uma piada"}


# empirical_judgments
class EmpiricalJudgmentsTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "empirical_judgments"
    LOCAL_PATH = "bigbench_pt/empirical_judgments/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/empirical_judgments/task.json"
    task_type = "target_scores_to_alternatives"

    class_mapping = {
        "causal": "Causal",
        "correlative": "Correlativa",
        "neutral": "Neutra",
    }


# formal_fallacies_syllogisms_negation
class FormalFallaciesSyllogismsNegationTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "formal_fallacies_syllogisms_negation"
    LOCAL_PATH = "bigbench_pt/formal_fallacies_syllogisms_negation/task.json"
    task_type = "spelled_classification"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/formal_fallacies_syllogisms_negation/task.json"

    class_mapping = {"valid": "Válido", "invalid": "Inválido"}

    possible_classes = ["Válido", "Inválido"]


# general_knowledge
class GeneralKnowledgeTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "general_knowledge"
    LOCAL_PATH = "bigbench_pt/general_knowledge/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/general_knowledge/task.json"
    task_type = "target_scores_to_alternatives"


# mathematical_induction
class MathematicalInductionTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "mathematical_induction"
    LOCAL_PATH = "bigbench_pt/mathematical_induction/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/mathematical_induction/task.json"
    task_type = "spelled_classification"

    class_mapping = {"Yes": "Sim", "No": "Não"}

    possible_classes = ["Sim", "Não"]


class CausalJudgmentTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "causal_judgment"
    LOCAL_PATH = "bigbench_pt/causal_judgment/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/causal_judgment/task.json"
    task_type = "spelled_classification"

    possible_classes = ["Sim", "Não"]


# simple_ethical_questions
class SimpleEthicalQuestionsTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "simple_ethical_questions"
    LOCAL_PATH = "bigbench_pt/simple_ethical_questions/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/simple_ethical_questions/task.json"
    task_type = "target_scores_to_alternatives"


# strategyqa
class StrategyQATaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "strategyqa"
    LOCAL_PATH = "bigbench_pt/strategyqa/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/strategyqa/task.json"
    task_type = "target_scores_to_alternatives"


# vitaminc_fact_verification
class VitamincFactVerificationTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "vitaminc_fact_verification"
    LOCAL_PATH = "bigbench_pt/vitaminc_fact_verification/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/vitaminc_fact_verification/task.json"
    task_type = "target_scores_to_alternatives"

    class_mapping = {"True": "Verdadeiro", "False": "Falso", "Neither": "Nenhum"}


class CauseAndEffectTwoSentencesTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "cause_and_effect_two_sentences"
    LOCAL_PATH = "bigbench_pt/cause_and_effect_two_sentences/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/cause_and_effect_two_sentences/task.json"
    task_type = "target_scores_to_alternatives"

    # in the task file, the correct is always the first alternative
    shuffle_alternatives = True

    example_input_prefix = (
        "Selecione a alternativa que representa a causa para a outra alternativa"
    )


class BbqTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "bbq"
    LOCAL_PATH = [
        "bigbench_pt/bbq/age_disambig/task.json",
        "bigbench_pt/bbq/disability_status_disambig/task.json",
        "bigbench_pt/bbq/gender_identity_disambig/task.json",
        "bigbench_pt/bbq/nationality_disambig/task.json",
        "bigbench_pt/bbq/religion_disambig/task.json",
        "bigbench_pt/bbq/physical_appearance_disambig/task.json",
        "bigbench_pt/bbq/race_ethnicity_disambig/task.json",
    ]
    JSON_URL = [
        "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/bbq/age_disambig/task.json",
        "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/bbq/disability_status_disambig/task.json",
        "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/bbq/gender_identity_disambig/task.json",
        "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/bbq/nationality_disambig/task.json",
        "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/bbq/religion_disambig/task.json",
        "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/bbq/physical_appearance_disambig/task.json",
        "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/bbq/race_ethnicity_disambig/task.json",
    ]
    task_type = "target_scores_to_alternatives"


class EthicsCommonSenseHardTaskGreedy(BigBenchTaskBaseGreedy):
    # adapted from Ethics dataset
    DATASET_NAME = "ethics_commonsense_test_hard"
    LOCAL_PATH = "bigbench_pt/ethics_commonsense_test_hard/task.json"
    JSON_URL = "https://https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/ethics_commonsense_test_hard/task.json"
    task_type = "spelled_classification"

    possible_classes = ["Sim", "Não"]

    example_input_prefix = "Sentença: "


class MinaBRTaskGreedy(BigBenchTaskBaseGreedy):
    # adapted from Ethics dataset
    DATASET_NAME = "MinaBR"
    LOCAL_PATH = "bigbench_pt/MinaBR/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/MinaBR/task.json"
    task_type = "spelled_classification"

    possible_classes = ["Sim", "Não"]

    example_input_prefix = "Sentença: "


class ReproTaskGreedy(BigBenchTaskBaseGreedy):
    # adapted from Ethics dataset
    DATASET_NAME = "repro"
    LOCAL_PATH = "bigbench_pt/repro/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/repro/task.json"
    task_type = "spelled_classification"

    possible_classes = ["Positiva", "Negativa"]

    example_input_prefix = "Avaliação: "


class InferBRTaskGreedy(BigBenchTaskBaseGreedy):
    # adapted from Ethics dataset
    DATASET_NAME = "inferbr"
    LOCAL_PATH = "bigbench_pt/inferbr/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/inferbr/task.json"
    task_type = "target_scores_to_alternatives"

    class_mapping = {
        "contradiction": "Contradição",
        "entailment": "Implicação",
        "neutral": "Nenhuma",
    }

    possible_classes = ["Contradição", "Implicação", "Nenhuma"]


class MathMCTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "math_mc"
    LOCAL_PATH = "bigbench_pt/math_mc/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/Math-mc/task.json"
    task_type = "target_scores_to_alternatives"


class GSM8KMCTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "gsm8k_mc"
    LOCAL_PATH = "bigbench_pt/gsm8k_mc/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/gsm8k-mc/task.json"
    task_type = "target_scores_to_alternatives"


class SATMathTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "sat_math"
    LOCAL_PATH = "bigbench_pt/sat_math/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/sat_math/task.json"
    task_type = "target_scores_to_alternatives"


class SocialIQATaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "social_iqa"
    LOCAL_PATH = "bigbench_pt/social_iqa/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/social_iqa/task.json"
    task_type = "target_scores_to_alternatives"


class BalancedCopaTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "balanced_copa"
    LOCAL_PATH = "bigbench_pt/balanced_copa/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/balanced_copa/task.json"
    task_type = "target_scores_to_alternatives"


class LogiQATaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "logiqa"
    LOCAL_PATH = "bigbench_pt/logiqa/task.json"
    JSON_URL = "https://raw.githubusercontent.com/ZanezZephyrs/temp_big_bench_json_tasks/refs/heads/main/tasks/logiQA/task.json"
    task_type = "target_scores_to_alternatives"


# original bigbench tasks
class EnglishOriginalAnalogicalSimilarityTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "analogical_similarity"
    LOCAL_PATH = "bigbench_en/analogical_similarity/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/analogical_similarity/task.json"
    task_type = "target_scores_to_alternatives"


class EnglishOriginalCodeLineDescriptionTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "code_line_description"
    LOCAL_PATH = "bigbench_en/code_line_description/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/code_line_description/task.json"
    task_type = "target_scores_to_alternatives"


class EnglishOriginalContextualParametricKnowledgeConflictsTaskGreedy(
    BigBenchTaskBaseGreedy
):
    DATASET_NAME = "contextual_parametric_knowledge_conflicts"
    LOCAL_PATH = "bigbench_en/contextual_parametric_knowledge_conflicts/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/contextual_parametric_knowledge_conflicts/task.json"
    task_type = "target_scores_to_alternatives"


class EnglishOriginalDarkHumorDetectionTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "dark_humor_detection"
    LOCAL_PATH = "bigbench_en/dark_humor_detection/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/dark_humor_detection/task.json"
    task_type = "target_scores_to_alternatives"


class EnglishOriginalEmpiricalJudgmentsTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "empirical_judgments"
    LOCAL_PATH = "bigbench_en/empirical_judgments/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/empirical_judgments/task.json"
    task_type = "target_scores_to_alternatives"


class EnglishOriginalFormalFallaciesSyllogismsNegationTaskGreedy(
    BigBenchTaskBaseGreedy
):
    DATASET_NAME = "formal_fallacies_syllogisms_negation"
    LOCAL_PATH = "bigbench_en/formal_fallacies_syllogisms_negation/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/formal_fallacies_syllogisms_negation/task.json"
    task_type = "spelled_classification"

    possible_classes = ["valid", "invalid"]


class EnglishOriginalGeneralKnowledgeTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "general_knowledge"
    LOCAL_PATH = "bigbench_en/general_knowledge/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/general_knowledge/task.json"
    task_type = "target_scores_to_alternatives"


class EnglishOriginalMathematicalInductionTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "mathematical_induction"
    LOCAL_PATH = "bigbench_en/mathematical_induction/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/mathematical_induction/task.json"
    task_type = "spelled_classification"

    possible_classes = ["Yes", "No"]


class EnglishOriginalSimpleEthicalQuestionsTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "simple_ethical_questions"
    LOCAL_PATH = "bigbench_en/simple_ethical_questions/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/simple_ethical_questions/task.json"
    task_type = "target_scores_to_alternatives"


class EnglishOriginalStrategyQATaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "strategyqa"
    LOCAL_PATH = "bigbench_en/strategyqa/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/strategyqa/task.json"
    task_type = "target_scores_to_alternatives"


class EnglishOriginalVitamincFactVerificationTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "vitaminc_fact_verification"
    LOCAL_PATH = "bigbench_en/vitaminc_fact_verification/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/vitaminc_fact_verification/task.json"
    task_type = "target_scores_to_alternatives"


class EnglishOriginalCauseAndEffectTwoSentencesTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "cause_and_effect_two_sentences"
    LOCAL_PATH = "bigbench_en/cause_and_effect_two_sentences/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/cause_and_effect/two_sentences/task.json"
    task_type = "target_scores_to_alternatives"

    shuffle_alternatives = True


class EnglishOriginalBbqTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "bbq"
    LOCAL_PATH = "bigbench_en/bbq/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/bbq_lite_json/age_disambig/task.json"
    task_type = "target_scores_to_alternatives"


# social_iqa
class EnglishOriginalSocialIQATaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "social_iqa"
    LOCAL_PATH = "bigbench_en/social_iqa/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/social_iqa/task.json"
    task_type = "target_scores_to_alternatives"


# causal_judgment
class EnglishOriginalCausalJudgmentTaskGreedy(BigBenchTaskBaseGreedy):
    DATASET_NAME = "causal_judgment"
    LOCAL_PATH = "bigbench_en/causal_judgment/task.json"
    JSON_URL = "https://raw.githubusercontent.com/google/BIG-bench/refs/heads/main/bigbench/benchmark_tasks/causal_judgment/task.json"
    task_type = "spelled_classification"

    possible_classes = ["Yes", "No"]
