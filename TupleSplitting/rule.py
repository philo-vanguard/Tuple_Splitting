class Predicate:
    def __init__(self):
        self.index1 = None
        self.index2 = None
        self.attr1 = ""
        self.attr2 = ""
        self.constant = ""
        self.operator = ""
        self.tableName = ""
        self.confidence = 0.0  # the confidence threshold for Mc rules
        self.type = ""  # an Enum {constant, non-constant, ML, M_c, M_d}

    def transform(self, str_predicate, op):
        if str_predicate.strip().startswith("t0.") or str_predicate.strip().startswith("t1."):
            info = str_predicate.strip().split(op)
            self.index1 = int(info[0].strip().split(".")[0][1])
            self.attr1 = info[0].strip().split(".")[1]
            self.operator = op
            if info[1].strip().startswith("t1."):
                self.index2 = int(info[1].strip().split(".")[0][1])
                self.attr2 = info[1].strip().split(".")[1]
                self.type = "non-constant"
            else:
                self.constant = eval(info[1].strip()) if "\'" in info[1] else info[1].strip()
                self.type = "constant"
        elif str_predicate.strip().startswith("Mc"):
            info = str_predicate.strip().split(op)
            self.type = "M_c"
            self.index1 = 0
            self.operator = op
            self.confidence = float(info[1].strip())
            barA_B = info[0].split(")")[0].split("(")[1]
            bar_A = barA_B.split("]")[0].split("[")[1].strip().split(",")  # bar_A attribute list
            self.attr1 = [i.strip() for i in bar_A]
            self.attr2 = barA_B.split("]")[1].split(",")[1].strip().split(".")[1]  # B attribute

    def get_index1(self):
        return self.index1

    def get_index2(self):
        return self.index2

    def get_attr1(self):
        return self.attr1

    def get_attr2(self):
        return self.attr2

    def get_constant(self):
        return self.constant

    def get_operator(self):
        return self.operator

    def get_type(self):
        return self.type

    def get_confidence(self):
        return self.confidence

    def print_predicate(self):
        output = ""
        if self.type == "non-constant":
            output = "t" + str(self.index1) + "." + self.attr1 + " = t" + str(self.index2) + "." + self.attr2
        elif self.type == "constant":
            output = "t" + str(self.index1) + "." + self.attr1 + " = " + self.constant
        elif self.type == "Mc":
            output = "Mc(t" + str(self.index1) + ".[" + ", ".join(self.attr1) + "], t" + str(self.index1) + "." + self.attr2 + ") > " + str(self.confidence)
        return output


class REELogic:
    def __init__(self):
        self.tuple_variable_cnt = 0  # the number of tuples involved in this ree
        self.X = []  # a set of predicates, each predicate is a triple in the form of "t[A] operator t'[A]"
        self.e = []  # a triple in the form of "t[A] operator val"
        self.support = None
        self.confidence = ''  # the confidence of this REE
        self.type = "logic"  # an Enum {logic, KG, M_d, M_c}
        self.distinct_attributes_in_X = set()  # the distinct attributes in preconditions X
        self.relation = set()  # relation name
        self.currents = []  # the predicates in X
        self.RHS = None  # the predicate of Y

    # identify precondition X and consequence e based on textual rule
    def load_X_and_e(self, textual_rule):
        predicates = textual_rule.split(':', 1)
        predicates = predicates[1]  # remove "Rule:"
        predicates = predicates.split('->')
        precondition = predicates[0]  # obtain the predicates prior to symbol "->"
        consequence = predicates[1]  # obtain the predicates after symbol "->"

        if '⋀' in precondition:
            precondition = precondition.split('⋀')
        elif '^' in precondition:
            precondition = precondition.split('^')
        for predicate in precondition:
            predicate = predicate.strip()
            operator = self.obtain_operator(predicate)
            p = []
            if operator != '':  # if operator is one of <>, >=, <=, =, > and <
                pre = Predicate()
                pre.transform(predicate, operator)
                self.currents.append(pre)

                predicate = predicate.split(operator)
                p.append(predicate[0].strip())
                self.distinct_attributes_in_X.add(predicate[0].strip().split('.')[1])  # add the attribute name in distinct_attributes_in_X
                p.append(operator.strip())
                p.append(predicate[1].strip())
                self.X.append(p)

            else:  # no operator, e.g., airports(t0)
                self.tuple_variable_cnt += 1
                self.relation.add(predicate.split('(')[0])

        # obtain consequence e
        consequence = consequence.split(',')
        operator = self.obtain_operator(consequence[0])
        self.RHS = Predicate()
        self.RHS.transform(consequence[0], operator)
        self.support = float(consequence[1].split(':')[1].strip())  # the suport of this REE
        self.confidence = float(consequence[2].split(':')[1].strip())  # the confidence of this REE

        consequence = consequence[0]
        operator = self.obtain_operator(consequence)
        if operator != '':
            consequence = consequence.split(operator)
        p = []
        p.append(consequence[0].strip())
        if len(consequence) > 1:
            p.append(operator.strip())
            p.append(consequence[1].strip())
        self.e = p

    # identify the operator from <>, >=, <=, =, > and <
    def obtain_operator(self, predicate):
        operator = ''
        if (predicate.find('<>') != -1):
            operator = '<>'
        elif (predicate.find('>=') != -1):
            operator = '>='
        elif (predicate.find('<=') != -1):
            operator = '<='
        elif (predicate.find('=') != -1):
            operator = '='
        elif (predicate.find('>') != -1):
            operator = '>'
        elif (predicate.find('<') != -1):
            operator = '<'
        return operator

    def get_tuple_variable_cnt(self):
        return self.tuple_variable_cnt

    def get_X(self):
        return self.X

    def get_distinct_attributes_in_X(self):
        return self.distinct_attributes_in_X

    def get_e(self):
        return self.e

    def get_support(self):
        return self.support

    def get_confidence(self):
        return self.confidence

    def get_type(self):
        return self.type

    def get_currents(self):
        return self.currents

    def get_RHS(self):
        return self.RHS

    # print info for debug purpose
    def print_info(self):
        print('---------the number of tuple variable is ----------\n')
        print(self.tuple_variable_cnt)
        print('\n')
        print('---------the information of X is ----------\n')
        print(self.X)
        print('\n')
        print('---------the information of e is ----------\n')
        print(self.e)
        print('\n')
        print('---------the information of confidence is ----------\n')
        print(self.confidence)
        print('\n')

    def print_rule(self):
        output = ""

        output += self.currents[0].print_predicate()
        for idx in range(1, len(self.currents)):
            output += " ^ "
            output += self.currents[idx].print_predicate()

        output += " -> "
        output += self.RHS.print_predicate()

        return output


class REEMc:
    def __init__(self):
        self.tuple_variable_cnt = 0  # the number of tuples involved in this ree
        self.confidence = 0.0  # the confidence of this REE
        self.type = "M_c"  # an Enum {logic, KG, M_d, M_c}
        self.relation = ""  # relation name
        self.currents = Predicate()  # the predicates in X
        self.RHS = False  # the predicate of Y

    def load_X(self, textual_rule):
        predicates = textual_rule.split(':', 1)
        predicates = predicates[1]  # remove "Rule:"
        predicates = predicates.split('->')
        precondition = predicates[0]  # obtain the predicates prior to symbol "->"
        consequence = predicates[1]  # obtain the predicates after symbol "->"

        if '^' in precondition:
            precondition = precondition.split('^')
        elif '⋀' in precondition:
            precondition = precondition.split('⋀')
        for predicate in precondition:
            operator = self.obtain_operator(predicate)
            if operator != '':  # if operator is one of <>, >=, <=, =, > and <
                self.currents.transform(predicate, operator)

            else:  # no operator, e.g., airports(t0)
                self.tuple_variable_cnt += 1
                self.relation = predicate.split('(')[0]

    def get_currents(self):
        return self.currents

    def get_type(self):
        return self.type

    # identify the operator from <>, >=, <=, =, > and <
    def obtain_operator(self, predicate):
        operator = ''
        if predicate.find('<>') != -1:
            operator = '<>'
        elif predicate.find('>=') != -1:
            operator = '>='
        elif predicate.find('<=') != -1:
            operator = '<='
        elif predicate.find('=') != -1:
            operator = '='
        elif predicate.find('>') != -1:
            operator = '>'
        elif predicate.find('<') != -1:
            operator = '<'
        return operator

    def print_rule(self):
        output = ""
        output += self.currents.print_predicate()
        output += " -> "
        output += str(self.RHS)
        return output
