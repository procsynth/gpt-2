from gpt2 import GPT2

gpt = GPT2()

def callback(text, task):
	print(text)

text = "Alice: You are one of my anxieties. I feel like you're going to abandon me because our relationship won't last long like this Alice: Good evening yes Robert: Don't bother with that Robert: We will have time to tear each other later Alice: So we will necessarily get there Alice: And so will start a cycle again Robert: you take your head too much Robert: My clear ideas Robert: We will see each other Robert: It will be cool "

gpt.generate(text, callback, temperature=0.95, top_k=40)

