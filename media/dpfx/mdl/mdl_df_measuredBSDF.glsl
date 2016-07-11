float sphereCapArea( float h )
{
  return( 2.0f * PI * h );
}

/*************************/
/* Barycentric filtering */
/*************************/

/*
// Compute barycentric coordinates (u, v, w) for
// point p with respect to triangle (a, b, c)
// Transcribed from Christer Ericson's Real-Time Collision Detection
void Barycentric(Point a, Point b, Point c, float &u, float &v, float &w)
{
    Vector v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = Dot(v0, v0);
    float d01 = Dot(v0, v1);
    float d11 = Dot(v1, v1);
    float d20 = Dot(v2, v0);
    float d21 = Dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}

For a unit-triangle computing the barycentric coordinates is extremly efficient. Below is the computation for b to the right of a.
         c  
        /|
       / |
      /  |
     /   |
 v2 /    | v1
   /  p  |
  /      |
 /       |
/        |
a--------b
    v0

a = (0,0)
b = (1,0)
c = (1,1)

v0 = b - a = (1,0)
v1 = c - a = (1,1)
v2 = p - a = (p.x, p.y)

d00 = 1*1 + 0*0 = 1
d01 = 1*1 + 0*1 = 1
d11 = 1*1 + 1*1 = 2
d20 = p.x * 1 + p.y * 0 = p.x
d21 = p.x * 1 + p.y * 1 = p.x + py

denom = 1*2 - 1 = 1

v = 2*p.x - 1*(p.x + p.y) = 2*p.x - p.x - p.y = p.x - p.y
w = 1*(p.x + p.y) - 1*p.x = p.y
u = 1.0 - (p.x - p.y) - p.y = 1 - p.x
*/

vec4 clampedTexelFetch( in sampler3D mbsdf, in ivec3 uvw )
{
  ivec3 mbsdfSize = textureSize( mbsdf, 0 );
  // For an m*n texture the texels lie in the interval
  // [(0,0), (m-1, n-1)]. Clamp to this border.
  ivec3 tmp = clamp( uvw, ivec3(0), mbsdfSize - ivec3(1));
  return texelFetch( mbsdf, tmp, 0 );
}

vec3 sampleBSDFBarycentric( in sampler3D mbsdf, in vec3 uvw )
{
  // x -> Phi [0,180] deg
  // y -> ThetaOut [0,90) deg
  // z -> ThetaIn [0,90) deg
  
  // ThetaOut and ThetaIn are not specified for 90 deg. Repeat
  // the last sample point to extrapolate the missing value.
  // To do this map the y/z to the range [0, textureSize.yz]
  // and clamp to textureSize [0, textureSize.yz - 1]. This will
  // map the yz = (1,1) to the last texel.
  // Phi is specified for the whole range. Map x to the
  // interval [0, textureSize.x - 1] for sampling.
  
  ivec3 samplingInterval = textureSize( mbsdf, 0 ) - ivec3(1,0,0);
  
  uvw = clamp( uvw, 0, 1);
  
  vec3 texCoord = uvw * samplingInterval;
  ivec3 base = ivec3(texCoord);
  vec3 ta,tb,tc, weight;

  vec3 p = texCoord - base;

  // compute the barycentric coordinates for the point p in
  // the unit triangle. See *Barycentric filtering* for a
  // derivation of the formula.
  
  ivec3 texCoordA, texCoordB, texCoordC;
  if ( p.y > p.z )
  {
    weight.x = 1-p.y;
    weight.y = p.y-p.z;
    weight.z = p.z;
    
    texCoordB = base + ivec3(0,1,0);
  }
  else
  {
    weight.x = 1-p.z;
    weight.y = p.z-p.y;
    weight.z = p.y;
    texCoordB = base + ivec3(0,0,1);
  }
  
  texCoordA = base + ivec3(0,0,0);
  texCoordC = base + ivec3(0,1,1);

  ta = clampedTexelFetch( mbsdf, texCoordA ).rgb;
  tb = clampedTexelFetch( mbsdf, texCoordB ).rgb;
  tc = clampedTexelFetch( mbsdf, texCoordC ).rgb;
  
  vec3 rgb1 = weight.x * ta + weight.y * tb + weight.z * tc;

  ta = clampedTexelFetch( mbsdf, texCoordA + ivec3(1,0,0) ).rgb;
  tb = clampedTexelFetch( mbsdf, texCoordB + ivec3(1,0,0) ).rgb;
  tc = clampedTexelFetch( mbsdf, texCoordC + ivec3(1,0,0) ).rgb;

  vec3 rgb2 = weight.x * ta + weight.y * tb + weight.z * tc;
  
  vec3 rgb = (1-p.x) * rgb1 + p.x * rgb2;
  
  return rgb;
}

/*****************************/
/* Barycentric filtering end */
/*****************************/


vec4 mdl_df_measuredBSDF( in sampler3D mbsdf, in float multiplier, in int mode, in vec3 normal )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );

  float cosThetaIn  = dot(normal, lightDir);
  float cosThetaOut = dot(normal, viewDir);
  if ( 0.0f < cosThetaIn && 0.0f < cosThetaOut ) // viewDir and lightDir are in the hemisphere that normal defines. (Means MBSDFs only work for opaque materials.)
  {
    // due to limited floating point accuracy cosThetaIn might be > 1.0f; clamp
    cosThetaIn = min( cosThetaIn, 1.0f );
    cosThetaOut = min( cosThetaOut, 1.0f );

    // *.mbsdf file data order: 
    // brdf(theta_in, theta_out, phi) = measured_data[index_theta_in * (res_phi * res_theta) + index_theta_out * res_phi + index_phi];
    // That means indexing the sampler3D is with .x = phi, .y = theta_out, .z = theta_in
    vec3 uvw;
    uvw.z = acos( cosThetaIn )  * TWO_OVER_PI;
    uvw.y = acos( cosThetaOut ) * TWO_OVER_PI;

    vec3 projL = orthonormalize( normal, lightDir );
    vec3 projV = orthonormalize( normal, viewDir );
    
    // due to limited floating point accuracy cosThetaIn might be < -1.0f or > 1.0f; clamp
    float cosPhi = clamp( dot( projL, projV ), -1.0f, 1.0f );
    uvw.x = acos( cosPhi ) * ONE_OVER_PI;

    ivec3 mbsdfSize = textureSize( mbsdf, 0 );

#if 0
    // determine bsdf value by fetching the eight neighbours from the mbsdf sampler and slerp them
    uvw *= mbsdfSize;
    ivec3 uvwLow = ivec3( floor( uvw ) );
    ivec3 uvwHig = clamp( ivec3( ceil( uvw ) ), ivec3( 0 ), mbsdfSize - ivec3( 1 ) );

    vec4 t000 = texelFetch( mbsdf, ivec3( uvwLow.x, uvwLow.y, uvwLow.z ), 0 );
    vec4 t001 = texelFetch( mbsdf, ivec3( uvwLow.x, uvwLow.y, uvwHig.z ), 0 );
    vec4 t010 = texelFetch( mbsdf, ivec3( uvwLow.x, uvwHig.y, uvwLow.z ), 0 );
    vec4 t011 = texelFetch( mbsdf, ivec3( uvwLow.x, uvwHig.y, uvwHig.z ), 0 );
    vec4 t100 = texelFetch( mbsdf, ivec3( uvwHig.x, uvwLow.y, uvwLow.z ), 0 );
    vec4 t101 = texelFetch( mbsdf, ivec3( uvwHig.x, uvwLow.y, uvwHig.z ), 0 );
    vec4 t110 = texelFetch( mbsdf, ivec3( uvwHig.x, uvwHig.y, uvwLow.z ), 0 );
    vec4 t111 = texelFetch( mbsdf, ivec3( uvwHig.x, uvwHig.y, uvwHig.z ), 0 );

    float deltaTheta = PI_HALF / mbsdfSize.z;
    float t = uvw.z - uvwLow.z;
    float oneOverSin = 1.0f / sin( deltaTheta );
    float w0 = sin( ( 1.0f - t ) * deltaTheta ) * oneOverSin;
    float w1 = sin( t * deltaTheta ) * oneOverSin;
    vec4 t00 = w0 * t000 + w1 * t001;
    vec4 t01 = w0 * t010 + w1 * t011;
    vec4 t10 = w0 * t100 + w1 * t101;
    vec4 t11 = w0 * t110 + w1 * t111;

    deltaTheta = PI_HALF / mbsdfSize.y;
    t = uvw.y - uvwLow.y;
    oneOverSin = 1.0f / sin( deltaTheta );
    w0 = sin( ( 1.0f - t ) * deltaTheta ) * oneOverSin;
    w1 = sin( t * deltaTheta ) * oneOverSin;
    vec4 t0 = w0 * t00 + w1 * t01;
    vec4 t1 = w0 * t10 + w1 * t11;

    float deltaPhi = PI / mbsdfSize.x;
    t = uvw.x - uvwLow.x;
    vec3 tex = ( sin( ( 1.0f - t ) * deltaPhi ) * t0 + sin( t * deltaPhi ) * t1 ) / sin( deltaPhi ).rgb;
#elif 0
    vec3 tex = texture( mbsdf, uvw ).rgb;
#else
    vec3 tex = sampleBSDFBarycentric( mbsdf, uvw );
#endif

#if 0
    // solidAngle approximation by area of trapezoid spanned by deltaTheta and deltaPhi
    float deltaTheta = PI_HALF / mbsdfSize.z;
    float thetaCenter = ( 0.5f + floor( uvw.z * mbsdfSize.z ) ) * deltaTheta;

    float deltaPhi = PI / mbsdfSize.x;

    float solidAngle = deltaPhi * deltaTheta * sin( thetaCenter ) * cos( deltaTheta );
#elif 0
    // solidAngle calculation by dividing the spherical cap area by 2*mbsdfSize.x
    int iangleNView = int(uvw.z * mbsdfSize.z);
    int iangleNViewNext = iangleNView + 1;

    float angleNView = (float(iangleNView) * PI / 2.0f) / float(mbsdfSize.z);
    float angleNViewNext = (float(iangleNViewNext) * PI / 2.0f) / float(mbsdfSize.z);

    // compute the solid angle for the texel to sample to convert radiance to irradiance
    float solidAngle = (sphereCapArea( 1 - cos(angleNViewNext) ) - sphereCapArea( 1 - cos(angleNView) )) / (2 * mbsdfSize.x);
#else
    float solidAngle = 1;
#endif

    rgb = (multiplier * tex.rgb * lightDiffuse * solidAngle) * cosThetaIn;
  }
  return( vec4( rgb, 1.0f ) );
}

#if 1
unsigned int m_w; /* must not be zero, nor 0x464fffff */
unsigned int m_z;    /* must not be zero, nor 0x9068ffff */
 
uint get_random()
{
    m_z = 36969 * (m_z & 65535) + (m_z >> 16);
    m_w = 18000 * (m_w & 65535) + (m_w >> 16);
    return (m_z << 16) + m_w;  /* 32-bit result */
}
#endif

vec3 mdl_df_measuredBSDFEnvironmentRandom( in sampler3D mbsdf, in float multiplier, in int mode, in vec3 normal )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  vec3 rgbSum = vec3(0.0f);

  int totalSamples = 50;
  int samples = totalSamples;

  m_w = unsigned int(fract(varWorldPos.x) * 16777216);
  m_z = unsigned int(fract(varWorldPos.y) * 16777216);

  vec3 r = fract(varWorldPos);
  
  float gamma = 1.0;
  while ( samples > 0 )
  {
    // TODO negative values!
    r.x = float( float(get_random() & 0xffffff) / float(0xffffff));
    r.y = (float( float(get_random() & 0xffffff) / float(0xffffff)) - 0.5f) * 2.0f;
    r.z = (float( float(get_random() & 0xffffff) / float(0xffffff)) - 0.5f) * 2.0f;
    r.z = 0.0f;
    r.y = 1.0f - r.x;
    
    float l = length(r);
    if ( l <= 1.0f)
    {
      r /= vec3(l);
      
      lightDir = normalize(r.x * varNormal + r.y * varTangent + r.z * varBinormal);
      lightDiffuse = evalEnvironmentMap( lightDir, 0.0f );
      vec3 color = mdl_df_measuredBSDF( mbsdf, multiplier, mode, normal ).rgb;
      
      rgbSum += vec3(pow(color.x, 1.0f/gamma),pow(color.y, 1.0f/gamma),pow(color.z, 1.0f/gamma));
      //rgbSum += color;
      
      --samples;
    }
  }
  
  samples = totalSamples;
  rgb = PI * rgbSum / vec3(float(samples));
  //rgb = rgbSum;
  rgb = vec3(pow(rgb.x, 2.2f),pow(rgb.y, 2.2f),pow(rgb.z, 2.2f));
  return rgb;
}

vec3 mdl_df_measuredBSDFEnvironmentSimple( in sampler3D mbsdf, in float multiplier, in int mode, in vec3 normal )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  vec3 rgbSum = vec3(0.0f);

  // random sampling, not working yet

  int totalSamples = 50;
  int samples = totalSamples;

  m_w = unsigned int(fract(varWorldPos.x) * 16777216);
  m_z = unsigned int(fract(varWorldPos.y) * 16777216);

  vec3 r = fract(varWorldPos);
  
  //vec3 mainDir = normalize( (dot(viewDir, varTangent) * varTangent + dot(viewDir, varBinormal) * varBinormal) );
  vec3 mainDir = varBinormal;
  
  totalSamples = 0;
  float gamma = 1.0;
  for ( r.x = 0;r.x < 1.0; r.x += 0.05f )
  {
    r.y = 1.0f - r.x;
    r.z = 0.0f;

    r /= vec3(length(r));
    
    lightDir = normalize(r.y * varNormal + r.x * mainDir);
    lightDiffuse = evalEnvironmentMap( lightDir, 0.0f );
    vec3 color = mdl_df_measuredBSDF( mbsdf, multiplier, mode, normal ).rgb;
    
    rgbSum += vec3(pow(color.x, 1.0f/gamma),pow(color.y, 1.0f/gamma),pow(color.z, 1.0f/gamma));

    lightDir = normalize(r.y * varNormal - r.x * mainDir);
    lightDiffuse = evalEnvironmentMap( lightDir, 0.0f );
    color = mdl_df_measuredBSDF( mbsdf, multiplier, mode, normal ).rgb;
    
    rgbSum += vec3(pow(color.x, 1.0f/gamma),pow(color.y, 1.0f/gamma),pow(color.z, 1.0f/gamma));

    totalSamples += 2;
  }
  samples = totalSamples;
  rgb = 0.1 * rgbSum / vec3(float(samples));
  //rgb = rgbSum;
  rgb = vec3(pow(rgb.x, 2.2f),pow(rgb.y, 2.2f),pow(rgb.z, 2.2f));
  return rgb;
}

/** consine sampling **/
vec3 CosineSampleHemisphere()
{
  vec2 rnd;
  do {
    rnd.x = (float( float(get_random() & 0xffffff) / float(0xffffff)) - 0.5f) * 2.0f;
    rnd.y = (float( float(get_random() & 0xffffff) / float(0xffffff)) - 0.5f) * 2.0f;
  } while (length(rnd) > 1.0f);

  float r = sqrt(rnd.x);
  float theta = 2 * PI * rnd.y;
 
  float x = r * cos(theta);
  float y = r * sin(theta);
 
  return vec3(x, y, sqrt(max(0.0f, 1 - rnd.x)));
}

vec3 mdl_df_measuredBSDFEnvironmentRandomConsine( in sampler3D mbsdf, in float multiplier, in int mode, in vec3 normal )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  vec3 rgbSum = vec3(0.0f);

  int totalSamples = 50;
  int samples = totalSamples;

  m_w = unsigned int(fract(varWorldPos.x) * 16777216);
  m_z = unsigned int(fract(varWorldPos.y) * 16777216);

  vec3 r = fract(varWorldPos);
  
  float gamma = 1.0;
  while ( samples > 0 )
  {
    vec3 r = CosineSampleHemisphere();
    r = normalize(r);

    lightDir = r.z * varNormal + r.x * varTangent + r.y * varBinormal;
    lightDiffuse = evalEnvironmentMap( lightDir, 0.0f );
    vec3 color = mdl_df_measuredBSDF( mbsdf, multiplier, mode, normal ).rgb;

    rgbSum += vec3(pow(color.x, 1.0f/gamma),pow(color.y, 1.0f/gamma),pow(color.z, 1.0f/gamma));
    --samples;
  }
  
  samples = totalSamples;
  rgb = 100 * PI * rgbSum;
  //rgb = rgbSum;
  rgb = vec3(pow(rgb.x, 2.2f),pow(rgb.y, 2.2f),pow(rgb.z, 2.2f));
  return rgb;
}

vec3 mdl_df_measuredBSDFEnvironmentSpecular( in sampler3D mbsdf, in float multiplier, in int mode, in vec3 normal )
{
  lightDir = reflect( -viewDir, normal ); // DAR This means viewDir, normal, and lightDir are all in the same plane!
  lightDiffuse = evalEnvironmentMap( lightDir, 0.0f );
  return( mdl_df_measuredBSDF( mbsdf, multiplier, mode, normal ).rgb * 0.1f );
}

vec4 mdl_df_measuredBSDFEnvironment( in sampler3D mbsdf, in float multiplier, in int mode, in vec3 normal )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  vec3 rgbSum = vec3(0.0f);

  if ( sys_EnvironmentSamplerEnabled )
  {
  rgb = mdl_df_measuredBSDFEnvironmentSpecular( mbsdf, multiplier, mode, normal );
  //rgb = mdl_df_measuredBSDFEnvironmentRandom( mbsdf, multiplier, mode, normal );
  //rgb = mdl_df_measuredBSDFEnvironmentSimple( mbsdf, multiplier, mode, normal );
  //rgb = mdl_df_measuredBSDFEnvironmentRandomConsine( mbsdf, multiplier, mode, normal );
  }
  return( vec4( rgb, 1.0f ) );
}

 
