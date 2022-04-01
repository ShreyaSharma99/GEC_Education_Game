ERROR_TAGS = ['ArtOrDet', 'Mec', 'Nn', 'Npos', 'Others', 'Pform', 'Pref', 'Prep', 'Rloc-', 'SVA', 'Sfrag', 'Smod', 
              'Srun', 'Ssub', 'Trans', 'Um', 'V0', 'Vform', 'Vm', 'Vt', 'WOadv', 'WOinc', 'Wci', 'Wform', 'Wtone']

# standard_string = "missing or incorrect"
FEEDBACK_TEMPLATE = {'ArtOrDet' : "Article", 
                  'Mec' : "Punctuation / capitalization / spelling",
                  'Nn' : "Noun number incorrect",
                  'Npos' : "Possesive Noun",
                  'Pform' : "Pronoun Form",
                  'Pref' : "Pronoun reference",
                  'Prep' : "Preposition",
                  'Rloc-' : "Local Redundency",
                  'SVA' : "Subject-verb-agreement",
                  'Sfrag' : "Sentence fragmant",
                  'Smod' : "Dangling Modifier",  #modifier could be misinterpreted as being associated with a word other than the one intended
                  'Srun' : "Runons / comma splice",  # fault is the use of a comma to join two independent clauses
                  'Ssub' : "Subordinate clause", # "I know that Bette is a dolphin" - here "that Bette is a dolphin" occurs as the complement of the verb "know" 
                  'Trans' : "Conjuctions",
                  'V0' : "Missing verb",
                  'Vform' : "Verb form",
                  'Vm' : "Verb modal",
                  'Vt' : "Verb tense",
                  'WOadv' : "Adverb/adjective position",
                  'Wci' : " Wrong collocation", # it went fair well -> fairly
                  'Wform' : "Word form"
                    }