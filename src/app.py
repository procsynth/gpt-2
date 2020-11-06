import fire
import re
import os
from time import localtime, strftime
from gpt2 import GPT2

DEFAULT_CONV_LENGTH = 4096 # 2 part, 12 msg/h, 12h/day, 6day/week, 10 weeks 

PERF_WINDOW = 10


ALIASES = {
    "Bob" : "Robert",
    "Roger" : "Robert",
    "Caro" : "Carol",
    "Carolina" : "Carol",
    "Caroline" : "Carol"
}

BLOCKED_FROM = ['To', 'From', 'Subject', 'Sent', 'Report', "Guy"]
ALLOWED_FROM = ['Pyrrha', 'Alice', 'Robert', 'Carol', 'William', 'Arthur', 'Steve', 'Sarah']

with open('names.txt', 'r') as f:
    names = f.read().splitlines()
    ALLOWED_FROM += names


def gen_id():
    return strftime("%Y%m%d-%H%M%S", localtime())

def mean(array):
    r = 0
    for i in array:
        r += i
    return r / len(array)

def max_len(string):

    string = re.sub(r"(.)\1{2,}", r"\1\1", string)
    words = string.split(' ')
    max_l = 0
    for w in words:
        max_l = max(max_l, len(w))
    return max_l

def mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass


def append_log(file_name, raw, parsed, run_id):
    with open(file_name + "_raw.txt", 'a') as file:
        file.write("=" * 20 + " " + run_id + " " + "=" * 20 + "\n")
        file.write(raw+"\n")


    with open(file_name + ".txt", 'a') as file:
        file.write("=" * 20 + " " + run_id + " len:" + str(len(parsed))+" " + "=" * 20 + "\n")
        for m in parsed :
            file.write(m['from']+ ' \t— '+ m['content']+"\n")

        if len(parsed) == 0:
            file.write("no exploitable message\n")


class GPTApp():
    def __init__(self):
        self.messages = []
        self.gpt = GPT2()
        self.source_index = 0


    def start(
        self,
        source_file="sources.txt",
        conv_length=DEFAULT_CONV_LENGTH,
        nb_conv=1,
        temperature=0.95,
        top_k=40,
        result_dir="results__100",
        reinit=False
        ):

        with open(source_file, 'r') as f:
            self.sources = f.readlines()

        self.run_id = gen_id();

        self.perf_record = [80]*PERF_WINDOW
        self.perf_index = 0

        self.messages = self.parse(self.sources[self.source_index])

        mkdir(result_dir)
        self.result_dir = result_dir

        append_log(os.path.join(self.result_dir, self.run_id), self.sources[self.source_index], self.messages, self.run_id)

        self.gpt.generate(self.sources[self.source_index], self.handle_gen, temperature=temperature, top_k=top_k, run_id=self.run_id)

    def handle_gen(self, text, task):


        new_messages = self.parse(text)

        if len(new_messages) >= 0:
            self.messages += new_messages

            print("Batch length:", len(new_messages))
            append_log(os.path.join(self.result_dir, self.run_id), text, new_messages, task['run_id'])

        else:   
            print("no exploitable message") 
            append_log(os.path.join(self.result_dir, self.run_id), text, [], task['run_id'])


        # perf calculation

        self.perf_record[self.perf_index] = len(new_messages)
        self.perf_index = (self.perf_index + 1) % PERF_WINDOW

        perf = mean(self.perf_record)

        # detect text without spaces (indication of a mishape)
        max_l = max_len(text)

        print('='*80)
        print("perf : ", perf)
        print(self.perf_record)
        print("max_l :", max_l)

        next_prompt = ""

        for m in self.messages[-1*min(30, len(self.messages)):]:
            next_prompt += m['from']+": "+m['content']+" "


        if perf < 30 or max_l > 50:
            self.gpt.stop()
            self.gpt = GPT2()
            self.perf_record = [80]*PERF_WINDOW


        if len(self.messages) < DEFAULT_CONV_LENGTH and max_l <= 50:
            self.gpt.generate(next_prompt, self.handle_gen, temperature=task['temperature'], top_k=task['top_k'], run_id=gen_id())
        else:
            self.source_index = (self.source_index + 1) % len(self.sources)
            print(   )
            print(   )
            print(   )
            print('=-'*30 + "  NEW RUN  " + "-="*30)
            print(   )
            self.start()

    def parse(self, text):

        try:
            eot_index = text.index("<|endoftext|>")
            text = text[:eot_index]
            
        except Exception as e:
            pass

        try:
            text = text.replace("RAW Paste Data", "") 
        except Exception as e:
            pass


        text = re.sub(r" : ", ": ", text)

        parts = re.split(r'([A-z]+:)', text)


        for i, m in enumerate(parts):
            try:
                eot_index = m.index("\n")
                parts[i] = m[:eot_index] 
            except Exception as e:
                pass


        next_index = 0
        messages = []
        content_record = []

        while next_index < len(parts) - 1:
            if re.match(r'([A-z0-9]+:)', parts[next_index]) is not None:

                msg_from = parts[next_index][:-1]

                #  replace aliases
                if msg_from in ALIASES:
                    msg_from = ALIASES[msg_from]

                # filter non dialogue
                if msg_from not in ALLOWED_FROM:
                    next_index += 2
                    continue

                content = parts[next_index+1].strip()

                # remove duplicated chars, '[]' and () with no space inside
                content = re.sub(r"(.)\1{2,}", r"\1\1", content)
                content = re.sub(r"\[.+?\]", "", content)
                content = re.sub(r"\([^ ]+?\)", "", content)

                # filter duplicated messages
                if content in content_record[-7:-2]:
                    next_index += 2
                    continue



                content_record.append(content)                   

                messages.append({
                    "from" : msg_from,
                    "content" : content
                    })
                next_index += 2
            else:
                #ignore
                next_index += 1


        for m in messages:
            print(m['from'], '\t—', m['content'])

        return messages


if __name__ == '__main__':
    app = GPTApp()
    fire.Fire(app.start)
