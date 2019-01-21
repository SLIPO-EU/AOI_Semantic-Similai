
import json

source = "data/Topics/result_topics_1.txt"
target = "data/MyTopics/result_topics_1.txt"

# List of topics
topics = []
# load file
with open(source, 'r') as f:

    for line in f:

        # get topic-words of each topic
        topic = json.loads(line.split('|')[-1])
        # normalize frequency
        topic = {w: topic[w]/sum(topic.values()) for w in topic}
        # add to topics
        topics.append(topic)

# save topics
with open(target, 'w+') as f:
    json.dump(topics, f)    

