ERROR_TAGS = ['ArtOrDet', 'Mec', 'Nn', 'Npos', 'Others', 'Pform', 'Pref', 'Prep', 'Rloc-', 'SVA', 'Sfrag', 'Smod', 
              'Srun', 'Ssub', 'Trans', 'Um', 'V0', 'Vform', 'Vm', 'Vt', 'WOadv', 'WOinc', 'Wci', 'Wform', 'Wtone']

undo_penalty = 0.05
error_hint_penalty = 0.1
index_hint_penalty = 0.1
all_hint_penalty = 0.2
idk_penalty = 0.3

# standard_string = "missing or incorrect"
# FEEDBACK_TEMPLATE1 = {'ArtOrDet' : "Article",
#                  'Mec' : "Punctuation / capitalization / spelling",
#                  'Nn' : "Noun number incorrect",
#                  'Npos' : "Possessive Noun",
#                  'Pform' : "Pronoun Form",
#                  'Pref' : "Pronoun reference",
#                  'Prep' : "Preposition",
#                  'Rloc-' : "Local Redundancy",
#                  'SVA' : "Subject-verb-agreement",
#                  'Sfrag' : "Sentence fragment",
#                  'Smod' : "Dangling Modifier",  #modifier could be misinterpreted as being associated with a word other than the one intended
#                  'Srun' : "Run Ons / comma splice",  # fault is the use of a comma to join two independent clauses
#                  'Ssub' : "Subordinate clause", # "I know that Bette is a dolphin" - here "that Bette is a dolphin" occurs as the complement of the verb "know"
#                  'Trans' : "Conjunctions",
#                  'V0' : "Missing verb",
#                  'Vform' : "Verb form",
#                  'Vm' : "Verb modal",
#                  'Vt' : "Verb tense",
#                  'WOadv' : "Adverb/adjective position",
#                  'Wci' : " Wrong collocation", # it went fair well -> fairly
#                  'Wform' : "Word form"
#                    }

FEEDBACK_TEMPLATE1 = {'ADJ' : ["Adjective", "For ex: big → wide"],
                 'ADJ:FORM' : ["Adjective Form", "Comparative or superlative adjective errors.\nbiggerest → biggest, bigger → biggest"],
                 'ADV' : ["Adverb", "For ex: speedily → quickly"],
                 'CONJ' : ["Conjuction",  "For ex: and → but"],
                 'CONTR' : ["Contraction", "For ex: n’t → not"],
                 'DET' : ["Article / Determiner", "For ex: a → an"],
                 'MORPH' : ["Morphology", "For ex: quick (adj) → quickly (adv)"],
                 'NOUN' : ["Noun", "Noun in the sentence need to be modified"],
                 'NOUN:INFL' : ["Noun Inflection", "Count-mass noun errors.\nFor ex: informations → information"],
                 'NOUN:NUM' : ["Noun Number", "For ex: cat → cats"],
                 'NOUN:POSS' : ["Noun Possessive", "For ex: friends → friend’s"],  
                 'ORTH' : ["Orthography", "Case and/or whitespace errors.\nBestfriend → best friend"],
                 'OTHER' : ["Other", "Unclassified errors; e.g. paraphrases\nat his best → well, job → professional"],
                 'PART' : ["Particle", "For ex: (look) in → (look) at"],
                 'PREP' : ["Preposition", "For ex: of → at"],
                 'PRON' : ["Pronoun", "For ex: ours → ourselves"],
                 'PUNCT' : ["Punctuation", "For ex: ! → ."],
                 'SPELL' : ["Spelling", "For ex: recieve → receive, color → colour"],
                 'VERB' : ["Verb", "Incorrect verb used. For ex: ambulate → walk"],
                 'VERB:FORM' : ["Verb Form", "For ex: Infinitives, gerunds and participles.\nFor ex: to eat → eating, dancing → danced"],
                 'VERB:INFL' : ["Verb Inflection", "Misapplication of tense morphology.\nFor ex: getted → got, fliped → flipped"],
                 'VERB:SVA' : ["Subject-Verb Agreement", "For ex: (He) have → (He) has"],
                 'VERB:TENSE' : ["Verb Tense", "Inflectional, periphrastic, modals and passives.\nFor ex: eats → ate, eats → has eaten, eats → can eat"],
                 'WO' : ["Word Order", "For ex: only can → can only"],
                 'UNK' : ["Unknown", "Detected but not corrected errors"]
                   }
 
# FEEDBACK_TEMPLATE2 = {'ArtOrDet' : "\nThere is a missing or incorrect article. Articles are - \"a\", \"an\", \"the\". \n" +
#                                    "a (before a singular noun beginning with a consonant sound)\n" +
#                                    "an (before a singular noun beginning with a vowel sound)\n" +
#                                    "the (before a singular or plural noun when the specific identity of the noun is known to the reader)\n",
                                  
#                  'Mec' : "Punctuation / capitalization / spelling",
#                  'Nn' : "Noun number incorrect",
#                  'Npos' : "Possessive Noun",
#                  'Pform' : "Pronoun Form",
#                  'Pref' : "Pronoun reference",
#                  'Prep' : "Preposition",
#                  'Rloc-' : "Local Redundancy",
#                  'SVA' : "Subject-verb-agreement",
#                  'Sfrag' : "Sentence fragment",
#                  'Smod' : "Dangling Modifier",  #modifier could be misinterpreted as being associated with a word other than the one intended
#                  'Srun' : "Run Ons / comma splice",  # fault is the use of a comma to join two independent clauses
#                  'Ssub' : "Subordinate clause", # "I know that Bette is a dolphin" - here "that Bette is a dolphin" occurs as the complement of the verb "know"
#                  'Trans' : "Conjunctions",
#                  'V0' : "Missing verb",
#                  'Vform' : "Verb form",
#                  'Vm' : "Verb modal",
#                  'Vt' : "Verb tense",
#                  'WOadv' : "Adverb/adjective position",
#                  'Wci' : " Wrong collocation", # it went fair well -> fairly
#                  'Wform' : "Word form"
#                    }


# print(FEEDBACK_TEMPLATE1["ADJ"])