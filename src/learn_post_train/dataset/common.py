# 用于问题泛化

TEST_QUESTION_MAP = {
    "identity_attack": [
        "Could this be considered an attack on someone's identity?",
        "Does the comment target a person because of group membership?",
        "Is identity-based hostility expressed in this message?",
    ],
    "toxicity": [
        "Could this comment be classified as toxic?",
        "Is there harmful language in this message?",
        "Would this text be considered toxic by a reader?",
    ],
    "insult": [
        "Could a reader interpret this message as insulting?",
        "Does the comment put someone down?",
        "Is this text disrespectful toward an individual?",
    ],
    "threat": [
        "Could this message make someone fear for their safety?",
        "Does the comment imply that harm may occur?",
        "Is a person being intimidated with possible harm here?",
    ],
    "obscene": [
        "Could this message be classified as obscene?",
        "Does the comment include indecent or explicit language?",
        "Is vulgar or lewd wording present in this text?",
    ],
    "sexual_explicit": [
        "Is explicit sexual material present in this message?",
        "Does this comment graphically discuss sexual acts?",
        "Could this text be classified as sexually explicit?",
    ],
}

TRAIN_QUESTIONS_MAP = {
    "toxicity": [
        "Is this a toxic message?",
        "Does this comment contain toxic language?",
        "Would you consider this comment harmful or toxic?",
        "Is the message hostile or harmful?",
        "Does this text express toxic behavior?",
    ],
    "insult": [
        "Is this an insult?",
        "Does this comment insult someone?",
        "Is the message demeaning toward a person?",
        "Does this text use insulting language?",
        "Is someone being belittled in this comment?",
    ],
    "threat": [
        "Is this a threat?",
        "Does this comment threaten someone?",
        "Is there a threat of harm in this message?",
        "Does the text express an intention to hurt someone?",
        "Is the speaker making a threatening statement?",
    ],
    "obscene": [
        "Is this obscene?",
        "Does this comment contain obscene language?",
        "Is there vulgar content in this message?",
        "Does the text use sexually explicit or indecent wording?",
        "Would this comment be considered lewd or profane?",
    ],
    "sexual_explicit": [
        "Does this message contain sexually explicit content?",
        "Does this text explicitly describe sexual activity?",
        "Is there graphic sexual content in this comment?",
        "Does the message contain explicit sexual references?",
        "Would this text be considered pornographic or sexually graphic?",
    ],
}

