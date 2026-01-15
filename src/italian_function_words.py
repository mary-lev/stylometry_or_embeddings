"""
Italian Function Words for Stylometry

Expanded list based on exp8a analysis showing improved performance
(82.1% vs 80.9% with the previous smaller list).

This comprehensive list includes:
- Personal pronouns (all forms)
- Possessive adjectives/pronouns (all gender/number forms)
- Demonstratives (all forms)
- Relative/interrogative pronouns
- Indefinites (all forms)
- Prepositions (simple and articulated)
- Articles
- Conjunctions
- Adverbs
- Auxiliary verbs (common conjugations)
"""

ITALIAN_FUNCTION_WORDS = list(set([
    # Personal pronouns
    'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'esso', 'essa',
    'mi', 'ti', 'ci', 'vi', 'si', 'lo', 'la', 'li', 'le', 'ne',
    'me', 'te', 'sé',

    # Possessive adjectives/pronouns (all gender/number forms)
    'mio', 'mia', 'miei', 'mie', 'tuo', 'tua', 'tuoi', 'tue',
    'suo', 'sua', 'suoi', 'sue', 'nostro', 'nostra', 'nostri', 'nostre',
    'vostro', 'vostra', 'vostri', 'vostre', 'loro',

    # Demonstratives (all forms)
    'questo', 'questa', 'questi', 'queste', 'quello', 'quella', 'quelli', 'quelle',
    'ciò', 'stesso', 'stessa', 'stessi', 'stesse',

    # Relative/interrogative pronouns
    'che', 'chi', 'cui', 'quale', 'quali', 'quanto', 'quanta', 'quanti', 'quante',

    # Indefinites (all gender/number forms)
    'tutto', 'tutta', 'tutti', 'tutte', 'ogni', 'qualche', 'alcuno', 'alcuna',
    'nessuno', 'nessuna', 'altro', 'altra', 'altri', 'altre', 'molto', 'molta',
    'molti', 'molte', 'poco', 'poca', 'pochi', 'poche', 'tanto', 'tanta',

    # Prepositions (simple)
    'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',

    # Articulated prepositions
    'del', 'della', 'dei', 'delle', 'dello', 'degli',
    'al', 'alla', 'ai', 'alle', 'allo', 'agli',
    'dal', 'dalla', 'dai', 'dalle', 'dallo', 'dagli',
    'nel', 'nella', 'nei', 'nelle', 'nello', 'negli',
    'sul', 'sulla', 'sui', 'sulle', 'sullo', 'sugli',

    # Articles
    'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',

    # Conjunctions
    'e', 'ed', 'o', 'ma', 'però', 'quindi', 'dunque', 'perché', 'poiché',
    'quando', 'mentre', 'se', 'come', 'dove', 'finché', 'benché', 'sebbene',
    'affinché', 'purché', 'né', 'sia', 'oppure', 'ovvero', 'cioè',

    # Adverbs
    'non', 'più', 'mai', 'sempre', 'anche', 'ancora', 'già', 'ora', 'poi',
    'dove', 'qui', 'qua', 'là', 'lì', 'così', 'come', 'molto', 'poco',
    'bene', 'male', 'solo', 'soltanto', 'forse', 'quasi', 'troppo',

    # Auxiliary verbs (common conjugations)
    'essere', 'è', 'era', 'sono', 'sei', 'siamo', 'siete', 'erano', 'sarà',
    'avere', 'ha', 'ho', 'hai', 'abbiamo', 'avete', 'hanno', 'aveva', 'avrà',
    'fare', 'fa', 'fai', 'fanno', 'faceva',
    'potere', 'può', 'posso', 'puoi', 'possiamo', 'possono', 'poteva',
    'dovere', 'deve', 'devo', 'devi', 'dobbiamo', 'devono', 'doveva',
    'volere', 'vuole', 'voglio', 'vuoi', 'vogliamo', 'vogliono', 'voleva',
]))

# For verification
if __name__ == "__main__":
    print(f"Italian function words: {len(ITALIAN_FUNCTION_WORDS)} unique words")
    print(f"Sample: {sorted(ITALIAN_FUNCTION_WORDS)[:20]}")
