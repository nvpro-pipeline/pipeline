
#includes

#definitions
add_definitions(
  -DDP_SG_ALGORITHM_EXPORTS
)

#sources
set(SOURCES
  src/AnalyzeTraverser.cpp
  src/AppTraverser.cpp
  src/CombineTraverser.cpp
  src/DeindexTraverser.cpp
  src/DestrippingTraverser.cpp
  src/EliminateTraverser.cpp
  src/IdentityToGroupTraverser.cpp
  src/IndexTraverser.cpp
  src/Intersect.cpp
  src/ModelViewTraverser.cpp
  src/NormalizeTraverser.cpp
  src/Optimize.cpp
  src/OptimizeTraverser.cpp
  src/RayIntersectTraverser.cpp
  src/Replace.cpp
  src/Search.cpp
  src/SearchTraverser.cpp
  src/SmoothTraverser.cpp
  src/StatisticsTraverser.cpp
  src/StrippingTraverser.cpp
  src/Traverser.cpp
  src/TriangulateTraverser.cpp
  src/UnifyTraverser.cpp
  src/VertexCacheOptimizeTraverser.cpp
)

set(PRIVATE_HEADERS
)

set(PUBLIC_HEADERS
  Config.h
  AnalyzeTraverser.h
  AppTraverser.h
  ClipPlanesElement.h
  CombineTraverser.h
  DeindexTraverser.h
  DestrippingTraverser.h
  EliminateTraverser.h
  IdentityToGroupTraverser.h
  IndexTraverser.h
  Intersect.h
  ModelViewTraverser.h
  NormalizeTraverser.h
  Optimize.h
  OptimizeTraverser.h
  RayIntersectTraverser.h
  Replace.h
  Search.h
  SearchTraverser.h
  SmoothTraverser.h
  StatisticsTraverser.h
  StrippingTraverser.h
  TransformStack.h
  Traverser.h
  TriangulateTraverser.h
  UnifyTraverser.h
  VertexCacheOptimizeTraverser.h )

source_group(source FILES ${SOURCES})
source_group(header FILES ${PRIVATE_HEADERS})
source_group(""     FILES ${PUBLIC_HEADERS})

#target
add_library( DPSgAlgorithm OBJECT
  ${SOURCES}
  ${HEADERS}
  ${PUBLIC_HEADERS}
)

set_target_properties( DPSgAlgorithm PROPERTIES FOLDER "SG" )
