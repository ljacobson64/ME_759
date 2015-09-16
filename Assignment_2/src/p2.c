#include <stdio.h>
#include <string.h>

// Function which calculates the total length of all the arguments passed to a
// C program, not including white spaces
int get_total_length(int argc, char *argv[]) {
  int num_args = argc - 1;
  int total_length = 0;
  int i;

  // Special case where no arguments are passed
  if ( num_args == 0 ) return 0;

  // Sum the length of each argument
  for ( i = 0; i < num_args; i++ ) {
    total_length += strlen(argv[i+1]);
  }

  return total_length;
}

// Main function which calls the function get_total_length()
int main(int argc, char *argv[]) {
  int total_length = get_total_length(argc, argv);
  printf("%u\n", total_length);
  
  return 0;
}
