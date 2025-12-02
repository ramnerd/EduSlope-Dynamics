# history_data.py

def get_data():
    # 0 = Indus Valley, 1 = Mauryan Empire, 2 = British Raj
    texts = [
        "The Great Bath was a large public water tank in Mohenjo-daro.", # IVC
        "Seals with animal motifs like the unicorn were found in Harappa.", # IVC
        "The drainage system of the Indus cities was highly advanced.", # IVC
        "Ashoka the Great spread Dhamma across the Indian subcontinent.", # Maurya
        "Chanakya wrote the Arthashastra on statecraft and politics.", # Maurya
        "The Lion Capital of Ashoka was built in Sarnath.", # Maurya
        "The Battle of Plassey established Company rule in Bengal.", # British
        "The Railways were introduced by Lord Dalhousie in 1853.", # British
        "The partition of Bengal was announced by Lord Curzon." # British
    ]
    
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    label_map = {0: "Indus Valley Civilisation", 1: "Mauryan Empire", 2: "British Raj"}
    
    return texts, labels, label_map