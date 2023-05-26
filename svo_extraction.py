from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from nltk.tokenize import WhitespaceTokenizer

from transformers import BertTokenizer, RobertaTokenizer
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
PUNCT_WORDS = {'?', ',', '~', '@', ']', '"', '!', '*', '.', '[', '<', '|', '-', '(', '{', '\\', '^', '#', '`', '_', '$', ')', ';', '+', '&', '>', '=', '/', '%', ':', '}'}


def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs

def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs

def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs


def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False

def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False

def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs

def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs

def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None

def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None

def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated

def getAllObjs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))

    #potentialNewVerb, potentialNewObjs = getObjsFromAttrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs

def findSVOs(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    svos.append((sub.lower_, "!" + v.lower_ if verbNegated or objNegated else v.lower_, obj.lower_))
    return svos

def getAbuserOntoVictimSVOs(tokens):
    maleAbuser = {'he', 'boyfriend', 'bf', 'father', 'dad', 'husband', 'brother', 'man'}
    femaleAbuser = {'she', 'girlfriend', 'gf', 'mother', 'mom', 'wife', 'sister', 'woman'}
    neutralAbuser = {'pastor', 'abuser', 'offender', 'ex', 'x', 'lover', 'church', 'they'}
    victim = {'me', 'sister', 'brother', 'child', 'kid', 'baby', 'friend', 'her', 'him', 'man', 'woman'}

    svos = findSVOs(tokens)
    wnl = WordNetLemmatizer()
    passed = []
    for s, v, o in svos:
        s = wnl.lemmatize(s)
        v = "!" + wnl.lemmatize(v[1:], 'v') if v[0] == "!" else wnl.lemmatize(v, 'v')
        o = "!" + wnl.lemmatize(o[1:]) if o[0] == "!" else wnl.lemmatize(o)
        if s in maleAbuser.union(femaleAbuser).union(neutralAbuser) and o in victim:
            passed.append((s, v, o))
    return passed

def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])

def extractSVOs(nlp, bert_tokens, offset=1):

    cand_indexes = []
    tokens_word = []
    ### this is for bert implementation
    # for (i, token) in enumerate(bert_tokens):
    #     if (len(cand_indexes) >= 1 and token.startswith("##")):
    #         cand_indexes[-1].append(i)
    #         tokens_word[-1] = tokens_word[-1]+token.replace("##", "")
    #     else:
    #         cand_indexes.append([i])
    #         tokens_word.append(token)

    ### this is for roberta implementation
    for (i, token) in enumerate(bert_tokens):
        if (len(cand_indexes) >= 1 and not(token.startswith("Ġ"))) and not (token in PUNCT_WORDS):
            cand_indexes[-1].append(i)
            tokens_word[-1] = (tokens_word[-1]+token)
        else:
            cand_indexes.append([i])
            tokens_word.append(token.replace("Ġ", "").lower())

    svo_ids = []
    # nlp = English()
    tok_str = " ".join(tokens_word)
    if len(tok_str) == 0:
        return svo_ids
    tok = nlp(tok_str)
    svos = findSVOs(tok)

    # print(tokens_word)
    # print(bert_tokens)
    #printDeps(tok)
    #print(svos)

    for svo in svos:
        sub, vb, obj = svo
        if sub in tokens_word:
            sub_idx = tokens_word.index(sub)
        else:
            break
        if vb in tokens_word[sub_idx:]:
            vb_idx = tokens_word[sub_idx:].index(vb) + sub_idx
        elif vb.replace("!", "") in tokens_word[sub_idx:]:
            vb_idx = tokens_word[sub_idx:].index(vb.replace("!", "")) + sub_idx
        else:
            break
        if obj in tokens_word[vb_idx:]:
            obj_idx = tokens_word[vb_idx:].index(obj) + vb_idx
        else:
            break
        # triplets = [cand_indexes[sub_idx][0] + offset, cand_indexes[vb_idx][0] + offset, cand_indexes[obj_idx][0] + offset]
        triplets = [[cand_indexes[sub_idx][0] + offset, len(cand_indexes[sub_idx])], [cand_indexes[vb_idx][0] + offset, len(cand_indexes[vb_idx])], [cand_indexes[obj_idx][0] + offset, len(cand_indexes[obj_idx])]]
        svo_ids.append(triplets)
    return svo_ids

def getSVOIDs(svos, bert_tokens, offset=1):

    cand_indexes = []
    tokens_word = []
    for (i, token) in enumerate(bert_tokens):
        if (len(cand_indexes) >= 1 and token.startswith("##")):
            cand_indexes[-1].append(i)
            tokens_word[-1] = tokens_word[-1]+token.replace("##", "")
        else:
            cand_indexes.append([i])
            tokens_word.append(token)

    svo_ids = []
    #svos = [item for sublist in svos for item in sublist if len(item) > 0]
    svos = [item for item in svos if len(item) > 0]
    for svo in svos:
        sub, vb, obj = svo
        if sub in tokens_word:
            sub_idx = tokens_word.index(sub)
        else:
            break
        if vb in tokens_word[sub_idx:]:
            vb_idx = tokens_word[sub_idx:].index(vb) + sub_idx
        elif vb.replace("!", "") in tokens_word[sub_idx:]:
            vb_idx = tokens_word[sub_idx:].index(vb.replace("!", "")) + sub_idx
        else:
            break
        if obj in tokens_word[vb_idx:]:
            obj_idx = tokens_word[vb_idx:].index(obj) + vb_idx
        else:
            break
        triplets = [cand_indexes[sub_idx][0] + offset, cand_indexes[vb_idx][0] + offset, cand_indexes[obj_idx][0] + offset]
        svo_ids.append(triplets)
    return svo_ids


def testSVOs(test_text):
    nlp = spacy.load("en_core_web_sm")
    # nlp = English()

    tok_str = test_text
    tok = nlp(tok_str)
    svos = findSVOs(tok)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large", do_lower_case=True)
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokens = tokenizer.tokenize(tok_str)
    cand_indexes = []
    tokens_word = []
    for (i, token) in enumerate(tokens):
        if (len(cand_indexes) >= 1 and not(token.startswith("Ġ"))) and not (token in PUNCT_WORDS):
            cand_indexes[-1].append(i)
            tokens_word[-1] = (tokens_word[-1]+token)
        else:
            cand_indexes.append([i])
            tokens_word.append(token.replace("Ġ", "").lower())
    print(cand_indexes)
    print(tokens_word)
    print(tokens)
    print(svos)
    svo_ids = []
    for svo in svos:
        sub, vb, obj = svo
        if sub in tokens_word:
            sub_idx = tokens_word.index(sub)
        else:
            break
        if vb in tokens_word:
            vb_idx = tokens_word.index(vb)
        elif vb.replace("!", "") in tokens_word:
            vb_idx = tokens_word.index(vb.replace("!", ""))
        else:
            break
        if obj in tokens_word:
            obj_idx = tokens_word.index(obj)
        else:
            break
        triplets = [(cand_indexes[sub_idx][0], len(cand_indexes[sub_idx])), (cand_indexes[vb_idx][0], len(cand_indexes[vb_idx])), (cand_indexes[obj_idx][0], len(cand_indexes[obj_idx]))]
        svo_ids.append(triplets)
    print(svo_ids)
    return svo_ids



def main():
    testSVOs("Detective: Laser-printer drums are easily damaged, and any nick in a drum will produce a blemish of similar dimensions on each page produced by that printer. So in matching a blemish on a page with a nick on a drum, we can reliably trace a suspicious laser-printed document to the precise printer on which it was produced.")

if __name__ == "__main__":
    main()