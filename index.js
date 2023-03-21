var exec = require("child_process").exec;

exec("python koutu.py", (err, stdout, stderr) => {

  console.log(stdout)

});