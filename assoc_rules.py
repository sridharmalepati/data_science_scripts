import sys
import json
import Orange

if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        with open('keys.basket', 'w') as t:

            for line in f:
                keys = [k+v for k, v in json.loads(line).iteritems()]
                keys = [k for k in keys if k and not 'http:/' in k and len(k) < 20]

                try:
                    t.write(', '.join(keys))
                    t.write('\n')
                except UnicodeEncodeError:
                    print 'error'

    print 'computing'

    data = Orange.data.Table('keys.basket')
    rules = Orange.associate.AssociationRulesSparseInducer(data, support=0.1)

    print "%4s %4s  %s" % ("Supp", "Conf", "Rule")
    for r in rules[:25]:
        print "%4.1f %4.1f  %s" % (r.support, r.confidence, r)
