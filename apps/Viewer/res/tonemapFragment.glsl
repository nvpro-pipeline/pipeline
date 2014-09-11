void main()
{
  vec3 ldrColor = invWhitePoint * texture(tonemapHDR, varTexCoord0).rgb;
  ldrColor *= (ldrColor * burnHighlights + 1.0f) / (ldrColor + 1.0f);
  float greyscale = dot(ldrColor, vec3(0.176204f, 0.812985f, 0.0108109f)); 
  ldrColor = mix(vec3(greyscale), ldrColor, saturation);
  float intens = dot(ldrColor, vec3(0.176204f, 0.812985f, 0.0108109f));
  if (intens < 1.0f)
  {
    ldrColor = mix(pow(ldrColor, vec3(crushBlacks)), ldrColor, sqrt(intens));
  }
  Color = vec4(pow(ldrColor, vec3(invGamma)), 1.0f);
}
