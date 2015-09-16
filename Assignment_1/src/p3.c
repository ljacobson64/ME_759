#include <stdio.h>
#include <string.h>

// Function which calculates the length of the first argument passed to a C
// program
int get_arg_length(int argc, char *argv[]) {
  // argc includes the call to the executable itself so it must be decremented
  // by 1 to find the number of arguments
  int num_args = argc - 1;

  // If no arguments are passed, return 0
  if ( num_args == 0 ) return 0;

  // If at least one argument is passed, return the length of the first argument
  return strlen(argv[1]);
}

int main(int argc, char *argv[]) {
  // Determine the length of the first argument by calling the user-defined
  // function get_arg_length()
  int arg_length = get_arg_length(argc, argv);

  // Print the length of the first argument
  printf("%u\n", arg_length);

  return 0;
}
