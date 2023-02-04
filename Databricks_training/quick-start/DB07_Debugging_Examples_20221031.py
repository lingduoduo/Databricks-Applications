# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Debugging in Databricks notebooks
# MAGIC 
# MAGIC Notebooks run on Databricks Runtime 11.2 and above support [The Python Debugger](https://docs.python.org/3/library/pdb.html) (pdb).
# MAGIC 
# MAGIC Some examples of using pdb in a notebook:
# MAGIC - Use `%debug` to debug from the last exception. This is helpful when you run into an unexpected error and are trying to debug the cause (similar to `pdb.pm()`).
# MAGIC - Use `%pdb on` to automatically start the interactive debugger after exceptions (but before program terminates).
# MAGIC 
# MAGIC Note that when you use these commands, you must finish using the debugger before you can run any other cell. Here are a few ways to exit the debugger:
# MAGIC - `c` or `continue` to finish running the cell.
# MAGIC - `exit` to throw an error and stop code execution.
# MAGIC - Cancel the command by clicking `Cancel` next to the output box.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## `%debug` : Post-mortem debugging
# MAGIC To use `%debug` in Databricks notebooks:
# MAGIC 1. Run commands in the notebook until an exception is raised.
# MAGIC 2. Run `%debug` in a new cell. The debugger starts running in the output area of the cell.
# MAGIC 3. To inspect a variable, type the variable name in the input field and press **Enter**.  
# MAGIC 4. You can change context and perform other debugger tasks, like variable inspection, using these commands. For the complete list of debugger commands, see the [pdb documentation](https://docs.python.org/3/library/pdb.html). Type the letter and then press **Enter**.  
# MAGIC    - `n`: next line
# MAGIC    - `u`: move up 1 level out of the current stack frame
# MAGIC    - `d`: move down 1 level out of the current stack frame
# MAGIC 5. Exit the debugger using one of the methods described in the first cell of this notebook.
# MAGIC 
# MAGIC Below is an example following these steps using `%debug`.

# COMMAND ----------

class ComplexSystem1:
  def getAccuracy(self, correct, total):
    # ...
    accuracy = correct / total
    # ...
    return accuracy
  
class UserTest:
  def __init__(self, system, correct, total):
    self.system = system
    self.correct = correct
    self.total = 0 # incorrectly set total!
    
  def printScore(self):
    print(f"You're score is: {self.system.getAccuracy(self.correct, self.total)}")
  
test = UserTest(
  system = ComplexSystem1(),
  correct = 10,
  total = 100
)

test.printScore()


# COMMAND ----------

# MAGIC %debug

# COMMAND ----------

# MAGIC %md
# MAGIC ## `%pdb on` : Pre-mortem debugging
# MAGIC To use `%pdb on` in Databricks notebooks:
# MAGIC 1. Turn auto pdb on by running `%pdb on` in the first cell of your notebook.
# MAGIC 2. Run commands in the notebook until an exception is raised. The interactive debugger starts.
# MAGIC 3. To inspect a variable, type the variable name in the input field and press **Enter**.  
# MAGIC 4. You can change context and perform other debugger tasks, like variable inspection, using these commands. For the complete list of debugger commands, see the [pdb documentation](https://docs.python.org/3/library/pdb.html). Type the letter and then press **Enter**.  
# MAGIC    - `n`: next line
# MAGIC    - `u`: move up 1 level out of the current stack frame
# MAGIC    - `d`: move down 1 level out of the current stack frame
# MAGIC 5. Exit the debugger using one of the methods described in the first cell of this notebook.
# MAGIC 
# MAGIC Below is an example following these steps using `%pdb on`.

# COMMAND ----------

# MAGIC %pdb on

# COMMAND ----------

class ComplexSystem2:
  def getAccuracy(self, correct, total):
    # ...
    accuracy = correct / total
    # ...
    return accuracy

system = ComplexSystem2()

## test coverage
print("Tests")
print(system.getAccuracy(10, 100) == 0.1)
print(system.getAccuracy(10, 0), 1)
