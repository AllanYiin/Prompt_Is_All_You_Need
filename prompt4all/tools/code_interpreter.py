import abc
import io
import ast
import subprocess
from contextlib import redirect_stdout
import abc
import ast
import io
import subprocess
from contextlib import redirect_stdout


class Executor(abc.ABC):
    @abc.abstractmethod
    def execute(self, code: str) -> str:
        pass


class PythonExecutor(Executor):
    locals = {}

    def execute(self, code: str) -> str:
        logger.info("Executing Python code: {}", code)
        output = io.StringIO()

        # Parse the code into an AST.
        tree = ast.parse(code, mode="exec")

        try:
            # Redirect standard output to our StringIO instance.
            with redirect_stdout(output):
                for node in tree.body:
                    # Compile and execute each node.
                    exec(
                        compile(
                            ast.Module(body=[node], type_ignores=[]), "<ast>", "exec"
                        ),
                        None,
                        PythonExecutor.locals,
                    )

                    # If the node is an expression, print its result.
                    if isinstance(node, ast.Expr):
                        eval_result = eval(
                            compile(ast.Expression(body=node.value), "<ast>", "eval"),
                            None,
                            PythonExecutor.locals,
                        )
                        if eval_result is not None:
                            print(eval_result)
        except Exception as e:
            logger.error("Error executing Python code: {}", e)
            return str(e)

        # Retrieve the output and return it.
        return output.getvalue()


class CppExecutor(Executor):
    def execute(self, code: str) -> str:
        with open("script.cpp", "w") as f:
            f.write(code)
        try:
            subprocess.run(["g++", "script.cpp"], check=True)
            output = subprocess.run(
                ["./a.out"], capture_output=True, text=True, check=True
            )
            return output.stdout
        except subprocess.CalledProcessError as e:
            # Here we include e.stderr in the output.
            raise subprocess.CalledProcessError(e.returncode, e.cmd, output=e.stderr)


async def python_exec(code: str, language: str = "python"):
    """
    Exexute code. \nNote: This endpoint current supports a REPL-like environment for Python only.\n\nArgs:\n    request (CodeExecutionRequest): The request object containing the code to execute.\n\nReturns:\n    CodeExecutionResponse: The result of the code execution.
    Parameters: code: (str, required): A Python code snippet for execution in a Jupyter environment, where variables and imports from previously executed code are accessible. The code must not rely on external variables/imports not available in this environment, and must print a dictionary `{"type": "<type>", "path": "<path>", "status": "<status>"}` as the last operation. `<type>` can be "image", "file", or "content", `<path>` is the file path (not needed if `<type>` is "content"), `<status>` indicates execution status. Display operations should save output as a file with path returned in the dictionary. If tabular data is generated, it should be directly returned as a string. The code must end with a `print` statement.the end must be print({"type": "<type>", "path": "<path>", "status": "<status>"})
    """

    myexcutor = PythonExecutor()
    code_output = myexcutor.execute(code)
    print(f"REPL execution result: {code_output}")
    response = {"result": code_output.strip()}
    return response
