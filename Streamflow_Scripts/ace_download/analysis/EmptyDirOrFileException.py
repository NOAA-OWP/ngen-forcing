class EmptyDirOrFileException( Exception ):
  """Exception raised for empty input files or directories.
      Attributes:
      expression -- input expression in which the error occurred
      message -- explanation of the error
  """
  pass

#  def __init__(self, message):
#      self.expression = expression
#      self.message = message
