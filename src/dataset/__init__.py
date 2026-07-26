"""Assembly of the raw lake layers into modelling tables.

The lake (``store.lake`` + ``collect.*``) stores provider-native rows and deliberately never stores
a join — a joined layer is where lookahead contamination becomes invisible. This package is the
other half: it reads those raw layers and builds a modelling frame **on demand**, applying the
lookahead guard at assembly time so the rule is one auditable piece of code rather than a property
of how the data happened to be written.
"""
