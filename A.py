import re

sentence = "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."
sentence2 = "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

rules = [
    [re.compile(r"with all safe and great in our lives"), "safely and happily in our lives"],
    [re.compile(r"Today is our dragon boat festival"), "Today is our Dragon Boat festival"],
    [re.compile(r"in our Chinese culture, to celebrate it"), "in our Chinese culture, we celebrate it"],
    [re.compile(r"bit delay"), "a bit of a delay"],
    [re.compile(r"they really tried best"), "they really tried their best"],
    [re.compile(r"at recent days"), "in recent days"],
    [re.compile(r"for paper and cooperation"), "for the paper and our cooperation"],
    [re.compile(r"at recent days"), "in recent days"],
    [re.compile(r"I believe the team"), "I believe in the team"]
]

def reconstruct_sentence(sentence):

    reconstructed_sentence = sentence

    for i in range(len(rules)):

        reconstructed_sentence = apply_rule(i, reconstructed_sentence)

    return reconstructed_sentence

def apply_rule(i, original_sentence):

    pattern = rules[i][0]

    replacement = rules[i][1]

    sentence = re.sub(pattern, replacement, original_sentence)

    if sentence != original_sentence:

        print(f"{pattern.pattern} -> {replacement}")

    return sentence


rewritten1 = reconstruct_sentence(sentence)
rewritten2 = reconstruct_sentence(sentence2)

print("ORIGINAL :", sentence)
print("REWRITTEN:", rewritten1)

print("ORIGINAL :", sentence2)
print("REWRITTEN:", rewritten2)


with open("reconstructed1.txt", "w", encoding="utf-8") as f:
    f.write(rewritten1)

with open("reconstructed2.txt", "w", encoding="utf-8") as f:
    f.write(rewritten2)

print("The reconstructed sentences have been saved")