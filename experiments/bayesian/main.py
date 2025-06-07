from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Structure
model = DiscreteBayesianNetwork(
    [
        ("Trump_Announcement", "Attend_NATO"),
        ("Court_Rulings", "Legal_Constraints"),
        ("Indictment_Status", "Legal_Constraints"),
        ("Legal_Constraints", "Attend_NATO"),
        ("Political_Climate", "Attend_NATO"),
        ("Health_Status", "Attend_NATO"),
        ("Travel_Logistics", "Attend_NATO"),
    ]
)

# 2. CPDs (you'd fill these in from expert judgment or historical data)
# Priors should reflect uncertainty, e.g., 50/50, not be deterministic.
cpd_announce = TabularCPD("Trump_Announcement", 2, [[0.5], [0.5]])

# New parent: Court rulings influence legal constraints
cpd_courtrulings = TabularCPD("Court_Rulings", 2, [[0.5], [0.5]])

# New parent: Indictment status influences legal constraints
cpd_indictment = TabularCPD("Indictment_Status", 2, [[0.5], [0.5]])

# Updated Legal_Constraints CPD to be probabilistic
# P(LC=1 | CR, IS)
# CR=0, IS=0: 0.05
# CR=0, IS=1: 0.80
# CR=1, IS=0: 0.70
# CR=1, IS=1: 0.95
# The order of states for evidence is (CR=0,IS=0), (CR=0,IS=1), (CR=1,IS=0), (CR=1,IS=1)
cpd_legal = TabularCPD(
    "Legal_Constraints",
    2,
    [[0.95, 0.20, 0.30, 0.05], [0.05, 0.80, 0.70, 0.95]],
    evidence=["Court_Rulings", "Indictment_Status"],
    evidence_card=[2, 2],
)

cpd_climate = TabularCPD("Political_Climate", 2, [[0.5], [0.5]])
cpd_health = TabularCPD("Health_Status", 2, [[0.5], [0.5]])
cpd_travel = TabularCPD("Travel_Logistics", 2, [[0.5], [0.5]])

# Example Attend_NATO CPD: Use a Noisy‑OR combination of individual factor weights
import itertools
import numpy as np

# Define influence weights for each parent state
weights = {
    ("Trump_Announcement", 1): 0.7,
    ("Trump_Announcement", 0): 0.3,
    ("Legal_Constraints", 1): 0.1,
    ("Legal_Constraints", 0): 0.5,
    ("Political_Climate", 1): 0.6,
    ("Political_Climate", 0): 0.2,
    ("Health_Status", 1): 0.4,
    ("Health_Status", 0): 0.1,
    ("Travel_Logistics", 1): 0.5,
    ("Travel_Logistics", 0): 0.2,
}

state_names_parents = {
    "Trump_Announcement": [0, 1],
    "Legal_Constraints": [0, 1],
    "Political_Climate": [0, 1],
    "Health_Status": [0, 1],
    "Travel_Logistics": [0, 1],
}

values = []
for announcement, legal, climate, health, travel in itertools.product(
    state_names_parents["Trump_Announcement"],
    state_names_parents["Legal_Constraints"],
    state_names_parents["Political_Climate"],
    state_names_parents["Health_Status"],
    state_names_parents["Travel_Logistics"],
):
    # Compute P(No attend) = product of (1 - weight) for each factor
    p_no = 1.0
    for var, val in zip(
        [
            "Trump_Announcement",
            "Legal_Constraints",
            "Political_Climate",
            "Health_Status",
            "Travel_Logistics",
        ],
        [announcement, legal, climate, health, travel],
    ):
        w = weights.get((var, val), 0)
        p_no *= 1 - w
    # P(Yes) is the complement
    p_yes = 1 - p_no
    values.append([p_no, p_yes])

attend_values = np.array(values).T.tolist()

cpd_attend = TabularCPD(
    variable="Attend_NATO",
    variable_card=2,
    evidence=list(state_names_parents.keys()),
    evidence_card=[len(state_names_parents[var]) for var in state_names_parents],
    values=attend_values,
)

model.add_cpds(
    cpd_announce,
    cpd_courtrulings,
    cpd_indictment,
    cpd_legal,
    cpd_climate,
    cpd_health,
    cpd_travel,
    cpd_attend,
)
model.check_model()

infer = VariableElimination(model)

# Now you can query with partial or full evidence.
# The original evidence:
evidence = {
    "Trump_Announcement": 1,
    "Court_Rulings": 1,
    "Indictment_Status": 1,
    # "Legal_Constraints": 1, # This is now inferred from its parents
    "Political_Climate": 0,
    "Health_Status": 0,
    "Travel_Logistics": 0,
}

# Example query with partial evidence: What if we only know about the announcement?
# posterior_partial = infer.query(["Attend_NATO"], evidence={"Trump_Announcement": 1})
# print("With only announcement evidence:")
# print(posterior_partial)


posterior_full = infer.query(["Attend_NATO"], evidence=evidence)
print("With the full evidence set:")
print(posterior_full)
