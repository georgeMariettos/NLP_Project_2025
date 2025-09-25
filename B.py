from transformers import pipeline
import nltk


text2 = """During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets"""
text1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""


nltk.download('punkt')

sentences_text1 = nltk.sent_tokenize(text1)
sentences_text2 = nltk.sent_tokenize(text2)


models = [
    ("stanford-oval/paraphraser-bart-large", "version1"),
    ("tuner007/pegasus_paraphrase", "version2"),
    ("humarin/chatgpt_paraphraser_on_T5_base", "version3"),
]

def paraphrase(sentences, text_id):
    for model_name, version in models:
        paraphraser = pipeline("text2text-generation", model=model_name)
        paraphrased_sentences = []
        for s in sentences:
            out = paraphraser(
                s,
                do_sample=False,
                num_beams=5,
                num_beam_groups=5,
                max_new_tokens=80,
                diversity_penalty=0.2,
                num_return_sequences=1
            )
            paraphrased_sentences.append(out[0]["generated_text"])


        filename = f"text{text_id}_{version}.txt"

        with open(filename, "w", encoding="utf-8") as f:

            f.write("\n".join(paraphrased_sentences))

        print(f"Saved: {filename}")


with open("original1.txt", "w", encoding="utf-8") as f:
    f.write(text1)

with open("original2.txt", "w", encoding="utf-8") as f:
    f.write(text2)

paraphrase(sentences_text1, 1)
paraphrase(sentences_text2, 2)

print("The reconstructed texts have been saved")
