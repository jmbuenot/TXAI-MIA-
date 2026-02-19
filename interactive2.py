#############################
########## IMPORTS ##########
#############################

## This code is developed by Jose Maria Alonso-Moral and Pablo Miguel Perez-Ferreiro

import warnings
warnings.filterwarnings('ignore')

# Loading plot tool (for ploting fuzzy sets and rules)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import SVG, HTML, display
import seaborn as sns

# Loading csv package for reading data files
import pandas as pd

# Loading lib to support handling of confusion matrix
import numpy as np

# Loading lib to support handling of iterators
import itertools

# Loading some models from sklearn for performance comparison
import sklearn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_text

# Loading JFML library (connects Python and Java, needed to use JFML, a Java package that supports fuzzy sets and systems)
from py4j.java_gateway import JavaGateway
from py4jfml.Py4Jfml import Py4jfml

# Loading the SimpleNLG library (for natural language generation)
from simplenlg.framework import *
from simplenlg.lexicon import *
from simplenlg.realiser.english import *
from simplenlg.phrasespec import *
from simplenlg.features import *

# Loading FCFEXPGEN library
from json import load
import re
from copy import deepcopy 
from fcfexpgen.feature import Attribute
from fcfexpgen.simple_rule import Simple_Rule
from fcfexpgen.complex_rule import Complex_Rule
from fcfexpgen.factual_explanation import Factual_Explanation
from fcfexpgen.CF_explanation import CF_Explanation

# Opening the JVM and Server for accessing to JFML
gateway = JavaGateway()

# Loading the realiser and corpus
lexicon = Lexicon.getDefaultLexicon()
nlgFactory = NLGFactory(lexicon)
realiser = Realiser(lexicon)

#########################################
########## AUXILIARY FUNCTIONS ##########
#########################################

########## Plotting utilities ###########

# Plots an PNG file as an image
def plot_png_file(f):
  plt.figure(figsize=[15,10])
  img = mpimg.imread(f)
  plt.imshow(img)
  plt.axis('off')
  plt.show()

# Plots the Fuzzy Partitions of a Fuzzy Variable
def plot_fuzzy_variable(fvt):
  print(fvt)
  plt.title(fvt.getName())
  plt.ylabel('membership degree')
  plt.xlabel('input range')
  plt.axis([fvt.getDomainleft(), fvt.getDomainright(), 0, 1])
  labels = fvt.getTerms()
  c = 0
  warn = False
  for i in labels:
      pp = i.getParam()
      if (len(pp)==1):
        plt.bar(pp[0], 1, width=0.2)
        warn = True
      else:
        if (c==0):
          plt.plot([pp[0], pp[0], pp[len(pp) - 2], pp[len(pp) - 1]], [1, 1, 1, 0], label=i.getName())
        elif (c==len(labels)-1):
          plt.plot([pp[0], pp[1], pp[len(pp) - 1], pp[len(pp) - 1]], [0, 1, 1, 1], label=i.getName())
        else:
          if (len(pp)==3):
            plt.plot([pp[0], pp[1], pp[2]], [0, 1, 0], label=i.getName())
          elif (len(pp)==4):
            plt.plot([pp[0], pp[1], pp[2], pp[3]], [0, 1, 1, 0], label=i.getName())
      c = c+1
  
  if (warn != True):
    plt.grid(True)
  else:
    xvalues= range(1,len(labels)+1)
    plt.xticks(xvalues)
       
  plt.legend(labels)
  plt.show()

# Plots the Fuzzy Terms (labels in a Fuzzy Partition)
def plot_term(v,t,i):
  m = t.getMembershipFunction()
  if (m.getName()=="SingletonMembershipFunction"):
    plt.title(v.getName()+" is "+t.getName()+" ("+str(i)+")")
  else:
    mv=t.getMembershipValue(i)
    plt.title(v.getName()+" is "+t.getName()+" ("+str(mv)+")")

  plt.ylabel('membership degree')
  plt.xlabel('input range')
  plt.axis([v.getDomainleft(), v.getDomainright(), 0, 1])
  pp = t.getParam()
  warn = False
  if (m.getName()=="SingletonMembershipFunction"):
    plt.bar(pp[0], i, width=0.2)
    warn = True
  else:
    if (m.getName()=="TriangularMembershipFunction"):
      plt.plot([pp[0], pp[1], pp[2]], [0, 1, 0])
    else:
      plt.plot([pp[0], pp[1], pp[2], pp[3]], [0, 1, 1, 0])

    plt.axvline(x=i, ymin=0, ymax=mv, ls='--', color='r')

  if (warn):
    xvalues= range(1,len(v.getTerms())+1)
    plt.xticks(xvalues)

# Plots Histogram for the values of a feature
def plot_histogram(fvt,d,b):
  n_bins = b
  plt.title(fvt.getName())
  plt.ylabel('number of data instances')
  plt.xlabel('input range')
  lim = len(d)
  if (lim > 100):
    lim = len(d)/3

  plt.axis([fvt.getDomainleft(), fvt.getDomainright(),0,lim])
  plt.hist(d, bins=n_bins, align='mid', rwidth=0.9)

  if (b < 10):
    xvalues= range(1,len(fvt.getTerms())+1)
    plt.xticks(xvalues)

  plt.show()

# Plots Fuzzy Rules in a Fuzzy Rule Base
def plot_fuzzy_rules(fr, opt):
  print("RULEBASE:")
  RBS = fr.getAllRuleBase()
  if (opt):
    print(RBS[0].getActivatedRules())
  else:
    print(RBS[0].toString())

  rules = RBS[0].getRules()
  for r in rules:
    if ( (opt==False) or (r.getEvaluation() > 0) ):
      print("Graphical view of " + r.toString())
      ccant = r.getAntecedent().getClauses()
      plt.figure(figsize=[15,5])
      k=1
      for a in ccant:
        va= a.getVariable()
        v= va.getValue()
        t= a.getTerm()
        plt.subplot(1,len(ccant)+1,k)
        plot_term(va,t,v)
        k=k+1

      cons = r.getConsequent().getThen().getClause()[0]
      cout= cons.getVariable()
      cv= cout.getValue()
      ct= cons.getTerm()
      plt.subplot(1,len(ccant)+1,k)
      plot_term(cout,ct,r.getEvaluation())
      plt.show()

# Plots a Confusion Matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    acc = []
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        if (i==j):
          acc += [cm[i, j]]

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    mm = round(np.mean(acc),2)
    print("Classification Ratio ("+title+"): Mean="+str(mm)+"; Stdev="+str(round(np.std(acc),2)))
    return mm;

# Plots the Pareto Front (Accuracy vs. Interpretability)
def plot_pareto_front(x,y,n,maxx):
  plt.title("Pareto Front")
  plt.ylabel('Accuracy (Classification Ratio)')
  plt.xlabel('Interpretability (Num of rules/leaves)')
  plt.axis([0, maxx, 0, 1])
  c=["ro","bo","go","rs","bs","gs","r*","b*","g*","r+","b+","g+"]
  for m in n:
    m_idx= n.index(m)
    plt.plot(x[m_idx], y[m_idx], c[m_idx], label=m)

  plt.grid(True)
  plt.legend()
  plt.show()

########## Evaluation tools ##########

# Performs cross-validation and analysis of Fuzzy Systems vs. ML models
def cross_val_analysis(n_folds, dfName, ml_model_names, ml_models, fis_model_names, fis_filenames, featureNames, target, nClasses):
  results = []

  for fold_id in range(n_folds):
    result = {}
    # load data
    fold_path = "testlib/"+str(dfName)+"/"+str(n_folds)+"CV/CV" + str(fold_id)
    trainingset_filename = dfName + ".txt.aux.train." + str(fold_id)
    testset_filename = dfName + ".txt.aux.test." + str(fold_id)

    trainingset = pd.read_csv(fold_path+"/"+trainingset_filename, header=None, names=featureNames+target, dtype={target[0]: str})
    testset = pd.read_csv(fold_path+"/"+testset_filename, header=None, names=featureNames+target, dtype={target[0]: str})

    # load Fuzzy Systems
    fis_models = []
    for fn in fis_filenames:
      fis = Py4jfml.load(fold_path+"/"+fn)
      fis_models.append(fis)

    # test FIS
    fis_inferences_array = []
    fis_n_rules_array = []
    for fis in fis_models:
      inferences = []
      for instance in range(len(testset)):
        for feature in featureNames:
             fuzzy_variable = fis.getVariable(feature)
             if fuzzy_variable is not None:
               actual_variable_value = testset[feature][instance]
               if (actual_variable_value > fuzzy_variable.getDomainright()):
                  actual_variable_value = fuzzy_variable.getDomainright()
               elif (actual_variable_value < fuzzy_variable.getDomainleft()):
                  actual_variable_value = fuzzy_variable.getDomainleft()

               fuzzy_variable.setValue(actual_variable_value)

        fis.evaluate()
        inferences.append(str(fis.getVariable(target[0]).getValue()))

      fis_inferences_array.append(inferences)
      fis_n_rules_array.append(len(fis.getAllRuleBase()[0].getRules()))

    # train ml models
    ml_array = [model.fit(trainingset[featureNames].values, trainingset[target].values.ravel()) for model in ml_models]
    # the models who have a method to get their leave number use it; otherwise they get assigned double the leaves of the biggest model (fuzzy or otherwise)
    ml_n_rules_array = [ml_model.get_n_leaves() if hasattr(ml_model, "get_n_leaves") and callable(getattr(ml_model, "get_n_leaves")) else -1 for ml_model in ml_models]
    ml_n_rules_array = [round(2*max(ml_n_rules_array + fis_n_rules_array)) if x == -1 else x for x in ml_n_rules_array]

    # test ml models
    ml_actual = [model.predict(testset[featureNames].values) for model in ml_array]

    # Now we gather results
    result["desired"] = testset[target].values.ravel()

    # process FIS
    result["n_rules_fis"] = fis_n_rules_array

    cm_fis = []
    for fis_idx in range(len(fis_inferences_array)):
        actual = fis_inferences_array[fis_idx]
        cm = confusion_matrix(testset[target].values.ravel(), actual, labels=[str(instance+1)+".0" for instance in range(nClasses)])
        cm_fis.append(cm)

    result["cm_fis"] = cm_fis

    # process ML models
    result["n_rules_ml"] = ml_n_rules_array

    cm_ml = []
    for ml_idx in range(len(ml_actual)):
        actual = ml_actual[ml_idx]
        cm = confusion_matrix(testset[target].values.ravel(), actual, labels=[str(instance+1)+".0" for instance in range(nClasses)])
        cm_ml.append(cm)

    result["cm_ml"] = cm_ml

    results.append(result)

  # Compute Confusion Matrixes
  fis_grand = []
  fis_accuracy = []
  fis_interp = []
  for fis_idx in range(len(fis_model_names)):
    sm= round(np.mean([results[instance]["n_rules_fis"][fis_idx] for instance in range(len(results))]),2)
    ss= round(np.std([results[instance]["n_rules_fis"][fis_idx] for instance in range(len(results))]),2)
    print("Number of rules ("+fis_model_names[fis_idx]+"): Mean="+str(sm) + "; Stdev="+str(ss))
    fis_grand.append(
         sum([results[instance]["cm_fis"][fis_idx] for instance in range(len(results))]) /
         sum([len(results[instance]["desired"]) for instance in range(len(results))]))

    fis_accuracy.append(sum([fis_grand[fis_idx][instance, instance] for instance in range(nClasses)]))
    fis_interp.append(sm)

  ml_grand = []
  ml_accuracy = []
  ml_interp = []
  for ml_idx in range(len(ml_models)):
    nn=sum([results[instance]["cm_ml"][ml_idx] for instance in range(len(results))])
    dd=sum([len(results[instance]["desired"]) for instance in range(len(results))])
    ml_grand.append(nn/dd)
    ml_accuracy.append(sum([ml_grand[ml_idx][instance, instance] for instance in range(nClasses)]))
    sm= round(np.mean([results[instance]["n_rules_ml"][ml_idx] for instance in range(len(results))]),2)
    ss= round(np.std([results[instance]["n_rules_ml"][ml_idx] for instance in range(len(results))]),2)
    print("Number of leaves ("+ml_model_names[ml_idx]+"): Mean="+str(sm) + "; Stdev="+str(ss))
    ml_interp.append(sm)

  # Plot results
  model_acc = []
  model_type = fis_grand
  for k in range(len(model_type)):
    mm = plot_confusion_matrix(model_type[k], [str(instance+1) for instance in range(nClasses)], normalize=True, title=fis_model_names[k])
    model_acc.append(mm)

  model_type = ml_grand
  for k in range(len(model_type)):
    mm = plot_confusion_matrix(model_type[k], [str(instance+1) for instance in range(nClasses)], normalize=True, title=ml_model_names[k])
    model_acc.append(mm)

  model_int = fis_interp + ml_interp

  return model_int, model_acc

########## Generation of Factual/Counterfactual Explanations ##########

## this code is developed by Ilia Stepin

# Definition of functions required for factual and counterfactual generation
def getInputVariables(fis):  
    variables = fis.getVariables()
    input_variables = dict()
    for var in variables:
        term_name = [term.getName() for term in var.getTerms()]
        if var.isInput(): 
            input_variables[var.getName()] = term_name
    return input_variables

def getShapes(fis):
    variables = fis.getVariables()
    shapes = dict()
    for variable in variables:
        if variable.isInput():
            shapes[variable.getName()] = dict()
            for term in variable.getTerms():                    
                if term.getTriangularShape() is not None:
                    shape = term.getTriangularShape()
                    shapes[variable.getName()][term.getName()] = [shape.getParameter(1), shape.getParameter(2), shape.getParameter(3)]
                elif term.getTrapezoidShape() is not None: 
                    shape = term.getTrapezoidShape()
                    shapes[variable.getName()][term.getName()] = [shape.getParameter(1), shape.getParameter(2), shape.getParameter(3), shape.getParameter(4)]
                elif term.getSingletonShape() is not None: 
                    shape = term.getSingletonShape()
                    shapes[variable.getName()][term.getName()] = [shape.getParameter(1)]
    feature_intervals_from_fis = dict()
    for feature, values in shapes.items():
        set_of_extreme_values = set()
        for mf in shapes[feature].values():
            set_of_extreme_values = set_of_extreme_values.union(mf) 
        if (len(set_of_extreme_values) > 0):
          feature_intervals_from_fis[feature] = [min(set_of_extreme_values), max(set_of_extreme_values)]
        else:
          feature_intervals_from_fis[feature] = [-1, -1]
    return shapes, feature_intervals_from_fis

def parse_fuzzy_rule(list_of_rules):
    rules = dict()
    for rule_id, rule in list_of_rules.items():
        try:
            activation = float(re.search("\(([\.Ee\d\-\+]+)\)", rule).group(1))
        except ValueError:
            activation = 0.0
        weight = float(re.search("weight=([\d\.]+)", rule).group(1))
        if re.search("IF ([-\(\)\w_\s.d]+) THEN", rule).group(1).find(' AND ')>=0:
            ante = re.search("IF ([-\(\)\w_\s.d]+) THEN", rule).group(1).split(" AND ")
        else:
            ante = [re.search("IF ([-\(\)\w_\s.d]+) THEN", rule).group(1)]
        antecedent = []
        for condition in ante:
            subj_obj = re.search("([-\(\)\w_\s]+) IS ([-\(\)\w_\s]+)", condition)
            antecedent.append(Simple_Rule(subj_obj.group(1), "=", subj_obj.group(2)))
        conseq = re.search("THEN ([-\w_\.\s\']+) \[", rule).group(1)
        consequent = re.search("IS ([-\w_\.\s\']+)", conseq).group(1)
        rules[rule_id] = Complex_Rule(rule_id, antecedent, consequent, activation, weight)
    return rules

def extract_all_rules(fis, input_variables, output_classes):
    rulebase = fis.getAllRuleBase()[0]
    all_output_rules = dict()
    for rule in str(rulebase).splitlines()[1:]:
        temp_str = rule.split(':')[1]
        rule_id = temp_str.split(' - ')[0].strip()
        all_output_rules[rule_id] = temp_str.split(' - ')[1].strip()
    all_rules = parse_fuzzy_rule(all_output_rules)
    return all_rules

def classify(data_instance_features, true_class, features_from_fis, fis, all_rules, feature_intervals_from_fis, rule_matrix_type="nonbinary"):
    for feature, value in data_instance_features:
        if feature in set([var.getName() for var in fis.getVariables()]):
            fuzzy_temp_var = fis.getVariable(feature)
            fuzzy_temp_var.setValue(float(value))  
    fis.evaluate()
    activated_rulebase = fis.getInferenceResults()
    all_activated_output_rules = dict()
    for line in activated_rulebase.splitlines():
        if re.search("\(OUTPUT\):\s", line):
            temp_str = line.split('=')[1]
            classification_result = line.split('=')[1].strip()
        if re.search("RULE\s", line):
            temp_str = line.split(':')[1]
            rule_id = temp_str.split(' - ')[0].strip()
            all_activated_output_rules[rule_id] = temp_str.split(' - ')[1].strip()
        else: continue
    
    if not all_activated_output_rules:
        return None, None
    all_activated_rules = parse_fuzzy_rule(all_activated_output_rules)
    factual_explanation_candidate_rules = [(rule_id, rule.activation) for rule_id, rule in all_activated_rules.items()]
    sorted_factual_explanation_candidate_rules = sorted(factual_explanation_candidate_rules, key=lambda x: x[1], reverse=True)
    for rule_id, rule_activation in sorted_factual_explanation_candidate_rules:
        all_rules[rule_id].activation = rule_activation
    factual_explanation_rule_id = sorted_factual_explanation_candidate_rules[0][0]
    
    for rule_id, rule in all_rules.items():
        rule_row = list()
        for feature, values in features_from_fis.items():
            for value in values:
                fuzzy_var = fis.getVariable(feature)
                fuzzy_term = fuzzy_var.getTerm(value)
                mv = fuzzy_term.getMembershipValue(float(dict(data_instance_features)[feature]))
                if rule_matrix_type == "binary":
                    [rule_row.append(1) if rule.findFuzzySimpleRule(feature, value) else rule_row.append(0)]
                elif rule_matrix_type == "nonbinary":
                    [rule_row.append(mv) if rule.findFuzzySimpleRule(feature, value) else rule_row.append(0)]
        all_rules[rule_id].matrix_row = deepcopy(rule_row)
    return all_rules[factual_explanation_rule_id], activated_rulebase#, min_adjusted_data_instance, max_adjusted_data_instance

def generate_test_instance_vector(data_instance_features, fis, features_from_fis, rule_matrix_type="nonbinary"):
    test_instance_vector = list()
    for feature, values in features_from_fis.items():
        for value in values:
            fuzzy_var = fis.getVariable(feature)
            fuzzy_term = fuzzy_var.getTerm(value)
            mv = fuzzy_term.getMembershipValue(float(dict(data_instance_features)[feature]))
            test_instance_vector.append(mv)
    if rule_matrix_type == "binary":
        test_instance_vector = [alpha_cut(value) for value in test_instance_vector]
    return test_instance_vector

def alpha_cut(input_value, threshold=0.5):  
    return 0 if input_value < threshold else 1 
    
def bitwise_xor(x, y):  
    test_instance_vector_binarised = np.asarray([alpha_cut(value) for value in x])
    print("test vector binarised: ", test_instance_vector_binarised)
    print('y vector: ', y.astype(int))
    bitwise_xor_vector = np.bitwise_xor(test_instance_vector_binarised, y.astype(int))
    return (len([i for i in bitwise_xor_vector if i == 1])/len(bitwise_xor_vector))

def euclidean_distance(x, y):   
    return np.sqrt(np.sum((x - y) ** 2))

def calculate_distances(all_rules, vector, dist_type="euc"):
    for rule_id, rule in all_rules.items():
        if dist_type == "xor":
            all_rules[rule_id].dist = bitwise_xor(np.asarray(vector), np.asarray(rule.matrix_row))
        elif dist_type == "euc":
            all_rules[rule_id].dist = euclidean_distance(np.asarray(vector), np.asarray(rule.matrix_row))

def generate_factual_explanation(factual_explanation_rule, features_from_json, features_from_fis, linguistic_terms_known):        
    factual_explanation = Factual_Explanation(factual_explanation_rule, features_from_json, features_from_fis, linguistic_terms_known)
    return factual_explanation
    
def generate_cf_explanations(fact_expl, output_classes, all_rules, linguistic_terms_known, distance="euc"):
    
    def generate_list_of_potential_cfs():
        list_of_potential_cfs = dict()
        for cf_output_class in output_classes:
            if cf_output_class == fact_expl.factual_explanation_rule.consequent:
                continue
            list_of_potential_cfs[cf_output_class] = sorted([(rule_id,rule.dist) for rule_id, rule in all_rules.items() if cf_output_class == all_rules[rule_id].consequent], key=lambda x: x[1])
        return list_of_potential_cfs

    def generate_filtered_list_of_potential_cfs(list_of_potential_cfs):
        filtered_list_of_potential_cfs = dict()
        for output, list_of_cfs in list_of_potential_cfs.items():
            min_rule_dist = min(list_of_cfs, key = lambda t: t[1])    
            filtered_list_of_potential_cfs[output] = [(rule_id, dist) for rule_id, dist in list_of_cfs if dist == min_rule_dist[1]]
        return filtered_list_of_potential_cfs
    
    def generate_final_list_of_potential_cfs(filtered_list_of_potential_cfs):
        final_list_of_cfs = dict()
        for output, list_of_cfs in filtered_list_of_potential_cfs.items():
            final_list_of_cfs[output] = all_rules[list_of_cfs[0][0]]
        return final_list_of_cfs
    
    list_of_potential_cfs = generate_list_of_potential_cfs()
    filtered_list_of_potential_cfs = generate_filtered_list_of_potential_cfs(list_of_potential_cfs)
    final_list_of_cfs = generate_final_list_of_potential_cfs(filtered_list_of_potential_cfs)
        
    best_cf_explanations = dict()
    for output, rule in final_list_of_cfs.items():
        best_cf_explanations[output] = CF_Explanation(rule, linguistic_terms_known)

    return list_of_potential_cfs, final_list_of_cfs, best_cf_explanations
    
def generate_counterfactuals(input, fnames, true_class, xml_filename, json_filename, linguistic_terms_known):
    json_file = open(json_filename, "r")
    json_data = load(json_file)
    
    features_from_json = dict()
    for component in json_data["attributes"]:
        features_from_json[component["name"]] = Attribute(component["name"], component["properties"], component["interval"])

    output_class_id2name = {consequent["id"]:consequent["name"] for consequent in json_data["consequents"]}
    
    #print(features_from_json)
    fis = Py4jfml.load(xml_filename)
    shapes, feature_intervals_from_fis = getShapes(fis)
    features_from_fis = getInputVariables(fis)    
    all_rules = extract_all_rules(fis, features_from_fis, output_class_id2name.values())
    data_instance_features = []
    for k in range(len(input)):
      data_instance_features.append((fnames[k],input[k]))
    
    print("Test instance features:\n{}".format(data_instance_features))
    factual_explanation_rule, _ = classify(data_instance_features, true_class, features_from_fis, fis, all_rules, feature_intervals_from_fis)
    print('True class: {}\nPredicted class: {}'.format(output_class_id2name[true_class], factual_explanation_rule.consequent))
    
    test_instance_vector = generate_test_instance_vector(data_instance_features, fis, features_from_fis)
    calculate_distances(all_rules, test_instance_vector)
    factual_explanation = generate_factual_explanation(factual_explanation_rule, features_from_json, features_from_fis, linguistic_terms_known)
    if linguistic_terms_known:
        print('\nFactual explanation:\n\n{}'.format(factual_explanation.surface_realisation()))
    else:  
        approx_factual_expl = factual_explanation.generate_ling_approx_explanation(features_from_json, shapes)
        print('\nFactual explanation:\n\n{}'.format(factual_explanation.surface_realisation(approx_factual_expl)))
    
    list_of_potential_cfs, final_list_of_cfs, cf_explanations = generate_cf_explanations(factual_explanation, output_class_id2name.values(), all_rules, linguistic_terms_known)   # array of CF explanations
    best_approximated_counterfactuals = dict.fromkeys(output_class_id2name.keys())
    cf_rule_id = dict.fromkeys(output_class_id2name.keys())
    approx_confidence = dict.fromkeys(output_class_id2name.keys())
    print("\nCounterfactual explanations:\n")
    if linguistic_terms_known:
        for output_class, cf_expl in cf_explanations.items():
            cf_explanations[output_class].cf_explanation_rule.antecedent = {rule.feature: rule.value for rule in cf_explanations[output_class].cf_explanation_rule.antecedent}
            for fact_key, fact_value in factual_explanation.factual_explanation_rule.antecedent.items():
                if fact_key in list(cf_explanations[output_class].cf_explanation_rule.antecedent.keys()) and fact_value == cf_explanations[output_class].cf_explanation_rule.antecedent[fact_key]:
                    del cf_explanations[output_class].cf_explanation_rule.antecedent[fact_key]                    
            print(cf_explanations[output_class].surface_realisation())
    else:  
        for output_class, cf_expl in cf_explanations.items():
            best_approximated_counterfactuals[output_class] = cf_expl.generate_ling_explanations(features_from_json, shapes, output_class)
            for fact_key, fact_value in approx_factual_expl.items():
                if fact_key in list(best_approximated_counterfactuals[output_class].keys()) and fact_value[0] == best_approximated_counterfactuals[output_class][fact_key][0]:
                    del best_approximated_counterfactuals[output_class][fact_key]                    
            print(cf_explanations[output_class].surface_realisation(best_approximated_counterfactuals[output_class], output_class))