#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main() {
  // Declare 64-bit integer
  uint64_t id_int = 9037862332;

  // Declare 3-character string
  char id_str[3];

  // Convert the 64-bit integer to a string and fill id_str with the first 3
  // characters of the string. The size in the snprintf() function includes the
  // trailing NULL so it must be incremented by 1.
  snprintf(id_str, sizeof(id_str) + 1, "%llu", id_int);

  // Print the 3-character string
  printf("Hello! I'm student %s.\n", id_str);

  return 0;
}
