import csv
import os
import tensorflow as tf
import tokenization_sentencepiece as tokenization

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    
class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, delimiter="\t", quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class LivedoorProcessor(DataProcessor):
  """Processor for the livedoor data set (see https://www.rondhuit.com/download.html)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ['dokujo-tsushin', 'it-life-hack', 'kaden-channel', 'livedoor-homme', 'movie-enter', 'peachy', 'smax', 'sports-watch', 'topic-news']

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        idx_text = line.index('text')
        idx_label = line.index('label')
      else:
        guid = "%s-%s" % (set_type, i)
        text_a = tokenization.convert_to_unicode(line[idx_text])
        label = tokenization.convert_to_unicode(line[idx_label])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

class NewProcessor(DataProcessor):
  def __init__(self, labels, train_file_name='train.tsv', dev_file_name='dev.tsv', test_file_name='test.tsv'):
    self.train_file_name = train_file_name
    self.dev_file_name = dev_file_name
    self.test_file_name = test_file_name
    self.labels = labels #python list
  
  def get_train_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.train_file_name)), 'train')

  def get_dev_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.dev_file_name)), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.test_file_name)), "test")

  def get_labels(self):
    return self.labels

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        idx_text = line.index('text')
        idx_label = line.index('label')
      else:
        guid = "%s-%s" % (set_type, i)
        text_a = tokenization.convert_to_unicode(line[idx_text])
        label = tokenization.convert_to_unicode(line[idx_label])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples
  






