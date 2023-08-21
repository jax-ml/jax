#include <stdio.h>

#include <cstring>
#include "xla/pjrt/c/pjrt_c_api.h"

extern "C" {

// Does not pass ownership of returned PJRT_Api* to caller
const PJRT_Api* GetPjrtApi();

}
