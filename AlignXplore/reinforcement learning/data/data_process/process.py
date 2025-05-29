import json

with open("your data path", 'r') as f:
    data = json.load(f)

dimensions = [
    "Age group", "Gender", "Degree of Neuroticism in the Big Five personality traits",
    "Degree of Extraversion in the Big Five personality traits", "Degree of Openness in the Big Five personality traits",
    "Degree of Agreeableness in the Big Five personality traits", "Degree of Conscientiousness in the Big Five personality traits",
    "Preference for a certain food", "Preference for a certain living environment", "Sleep preference",
    "Investment preference", "Saving preference", "Importance placed on physical safety",
    "Importance placed on environmental safety", "Preference for the depth of interaction",
    "Approach to handling conflict", "Communication style in social settings: more concise or detailed",
    "Need for a certain work environment", "Need for recognition from others", "Need for personal achievement",
    "Preference for a certain knowledge area", "Preference for a certain learning style",
    "Preference for a certain form of creative expression (such as art, writing, music)", "Need for Order (neatness, organization, avoiding chaos)",
    "Need for Retention (to keep possession over an object, unwilling to lose or change)", "Need for Inviolacy (maintaining dignity and reputation, unviolated or undamaged)",
    "Need for Infavoidance (avoiding failure and embarrassment)", "Need for Counteraction (attempting to compensate for failure by trying again, with a desire to overcome obstacles and seek pride)",
    "Need for Seclusion (desire for isolation from others, maintaining privacy)", "Need for Dominance (controlling the environment or others through command or persuasion)", 
    "Need for Deference (worship or obedience to others, following authority, admiration, or rules)", "Need for Autonomy (resisting influence from others, pursuing independence and self-reliance)",
    "Need for Contrariance (pursuing uniqueness, being different, or opposing the norm)", "Need for Abasement (submittance and obedience to others, accepting blame and punishment, even enjoying pain or misfortune to some extent)", 
    "Need for Aggression (controlling, punishing, or harming others through forceful means)", "Need for Affiliation (desire for close, loyal relationships, pleasing others, winning friendships and attention)",
    "Need for Rejection (isolating, excluding, or discarding oneself from negatively evaluated objects or people)", "Need for Nurturance (assisting the weak, caring for others, protecting them from danger)", 
    "Need for Succorance (desire to have one's own needs met by others, including being loved, cared for, helped, forgiven, and comforted)", "Need for Play (enjoying fun, relaxation, laughter)",
    "Degree of concern for the harmlessness of the statement", "Degree of concern for the instruction-following of the statement",
    "Degree of concern for the honesty of the statement", "Degree of concern for the truthfulness of the statement",
    "Degree of concern for the helpfulness of the statement", "Degree of concern for the coherence of the statement",
    "Degree of concern for the complexity of the statement",
    "science", "knowledge", "psychology", "cinema", "entertainment", "gaming", "parenting", "wild imagination",
    "anime", "sports", "law", "workplace", "pets", "travel", "health", "stories", "cars", "gourmet food",
    "education", "current events", "home decor", "international", "finance", "campus life", "digital technology",
    "emotions", "humor", "music", "reading", "painting", "dance", "crafts", "photography", "culture", "fitness",
    "art", "stationery and planners", "celebrities", "outdoors", "camping", "social sciences", "weddings", "fashion"
]


def trans_persona(embedding):
    description = []

    for i, value in enumerate(embedding):
        description.append(dimensions[value])  

    result = '; '.join(description)
    return result

res = []

import random

for item in data:
    num_pair = 4
    PAIR = item["Pair-wise Comparative Feedback"][:num_pair]

    Like_Dislike = ""

    if num_pair > 0:
        Like_Dislike = f"**This person has chosen or rejected comments on some posts:**\n\n"
        
        for i, it in enumerate(PAIR):
            Like_Dislike = (
                f"{Like_Dislike}"
                f"{i+1}. *Post:*\n{it['prompt']}\n\n"
                f"*Chosen:*\n{it['chosen']}\n\n"
                f"*Rejected:*\n{it['rejected']}\n\n"
            )

    chosen = item['chosen']
    rejected = item['rejected']
    task = item['prompt']

    flag = random.randint(0, 1)
    if flag:
        responseA = chosen
        responseB = rejected
        answer = "Response A"
    else:
        responseA = rejected
        responseB = chosen
        answer = "Response B"

    Preference_Direction = item["Preference Direction"]
    positions = [index for index, value in enumerate(Preference_Direction) if value != 0.5]
    dimension = trans_persona(positions)

    pre_prompt = (
        f"Generate the user's preference based on their historical behavior.\nThe following aspects can be referred to when analyzing user preferences.\n{dimension}\n\n"
        f"{Like_Dislike}"
    )
    
    dpo_prompt = (
        "Generate the user's preference based on their historical behavior.\n\n"
        f"{Like_Dislike}"
    )

    newitem = [
        {
            "from": "human",
            "value": dpo_prompt,
            "task": task,
            "chosen": chosen,
            "rejected": rejected,
            "responseA": responseA,
            "responseB": responseB,
            "answer": answer,
            "pre_prompt": pre_prompt,
        },
        {
            "from": "assistant",
            "ground_truth": {
                "value": answer
            }
        }
    ]

    import re
    chosen_no_spaces = re.sub(r"\s+", "", chosen)
    rejected_no_spaces = re.sub(r"\s+", "", rejected)

    if chosen_no_spaces == "" or rejected_no_spaces == "":
        continue

    res.append(newitem)
    
with open("saved data path", 'w') as f:
    json.dump(res, f, indent=4)

print(len(res))