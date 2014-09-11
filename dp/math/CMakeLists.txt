project(DPMath)

add_definitions("-DDP_MATH_EXPORTS")

include_directories(${DP_HOME})

set(PUBLIC_HEADERS
  Beziernt.h
  Boxnt.h
  Config.h
  math.h
  Matmnt.h
  Planent.h
  Quatt.h
  Spherent.h
  Trafo.h
  Vecnt.h
)

#let cmake determine linker language
set(SOURCES
  src/Math.cpp
  src/Matmnt.cpp
  src/Quatt.cpp
  src/Trafo.cpp
)

#sse versions
set(PUBLIC_HEADERS_SSE
  sse/Matmnt.h
  sse/Mat44f.h
  sse/Vecnt.h
  sse/Vec4f.h
)

#neon versions
set(PUBLIC_HEADERS_NEON
  neon/Matmnt.h
  neon/Mat44f.h
  neon/Vecnt.h
  neon/Vec4f.h
)

source_group(sources FILES ${SOURCES})
source_group("" FILES ${PUBLIC_HEADERS})
source_group("sse" FILES ${PUBLIC_HEADERS_SSE})
source_group("neon" FILES ${PUBLIC_HEADERS_NEON})

add_library(DPMath SHARED
  ${PUBLIC_HEADERS}
  ${PUBLIC_HEADERS_SSE}
  ${PUBLIC_HEADERS_NEON}
  ${SOURCES}
)

if (WIN32)
  target_link_libraries( DPMath DPUtil )
endif()
