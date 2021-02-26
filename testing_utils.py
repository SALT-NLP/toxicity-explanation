import pandas as pd

def category_split(categories, row):
  categories = categories.replace(' ', '')
  categories = categories.split(sep='><')
  if len(categories) != 4 or categories[0][0] != '<' or categories[3][-1] != '>':
    return True
  
  for i in range(3):
    categories[i] += '>'
    categories[i+1] = '<' + categories[i+1]

  row.extend(categories)
  return False

def append_to_df(df, row, col_names):
  row_df = pd.DataFrame([row], columns=col_names)
  df = df.append(row_df, ignore_index=True)
  return df

def predict_samples(model, tokenizer, actual, pred_col, active_test):
  empty_row = ['' for _ in range(len(pred_col) - 1)]
  pred = pd.DataFrame(columns=pred_col)
  bad_output = 0
  bad_categories = 0
  errors = 0
  error_inputs = []

  for i,post in enumerate(list(actual['post'])):
    if not(i % 100):
      print(i)

    try:
      encoded_post = tokenizer(post, return_tensors='pt')
      output = model.generate(encoded_post['input_ids'], \
                              max_length=150, \
                              eos_token_id=tokenizer.eos_token_id)
      output_str = tokenizer.decode(output[0])
    except:
      errors += 1
      error_inputs.append(post)
      pred = append_to_df(pred, [post] + empty_row, pred_col)
      continue

    output_list = output_str.split(sep=tokenizer.sep_token)
    if len(output_list) != 5:
      bad_output += 1
      pred = append_to_df(pred, [post] + empty_row, pred_col)
      continue

    new_row = []
    new_row.append(output_list[0].strip())
    bad_split = category_split(output_list[1], new_row)

    if bad_split:
      bad_categories += 1
      pred = append_to_df(pred, [post] + empty_row, pred_col)
      continue

    new_row.append(output_list[2].strip())
    new_row.append(output_list[3].strip())
    new_row.append(output_list[4][:-len(tokenizer.eos_token)])
    pred = append_to_df(pred, new_row, pred_col)

  actual.to_csv(active_test['TO ACTUAL'], index=False)
  pred.to_csv(active_test['TO PRED'], index=False)

  print("Errors: ", errors)
  print("Error Tuples: ", error_inputs)
  print("Bad Output: ", bad_output)
  print("Bad Categories: ", bad_categories)


def f1_score(actual, pred, col, pos, neg):
  tp = pred[(pred[col] == pos) & (pred[col] == actual[col])].shape[0]
  fp = pred[(pred[col] == pos) & (pred[col] != actual[col])].shape[0]
  tn = pred[(pred[col] == neg) & (pred[col] == actual[col])].shape[0]
  fn = pred[(pred[col] == neg) & (pred[col] != actual[col])].shape[0]
  
  if tp + fp == 0:
    precision = 0
  else:
    precision = tp / float(tp + fp)

  if tp + fn == 0:
    recall = 0
  else:
    recall = tp / float(tp + fn)
  
  if precision + recall == 0:
    f1 = 0
  else:
    f1 = 2 * ((precision*recall) / (precision + recall))
  
  return f1, precision, recall

def accuracy(actual, pred, col):
  match = pred[pred[col] == actual[col]].shape[0]
  total = pred.shape[0]
  return match / float(total)

