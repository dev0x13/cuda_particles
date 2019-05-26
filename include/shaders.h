#define STRINGIFY(A) #A

struct Shaders {

    const char* mblurVS = STRINGIFY(
            uniform float timestep;
            uniform float pointRadius;
            uniform float pointScale;
            void main() {
                vec3 pos = gl_Vertex.xyz;
                vec3 vel = gl_MultiTexCoord0.xyz;
                vec3 pos2 = (pos - vel * timestep).xyz;                 // previous position

                gl_Position = gl_ModelViewMatrix * vec4(pos, 1.0);   // eye space
                gl_TexCoord[0] = gl_ModelViewMatrix * vec4(pos2, 1.0);

                //gl_PointSize = 10;

                // aging
                float lifetime = gl_MultiTexCoord0.w;
                float age = gl_Vertex.w;
                float phase = (lifetime > 0.0) ? (age / lifetime) : 1.0;  // [0, 1]

                gl_TexCoord[1].x = phase;
                float fade = 1.0 - phase;
                //  float fade = 1.0;

                gl_FrontColor = vec4(gl_Color.xyz, gl_Color.w * fade);
            }
    );

    // vertex shader
    const char *mblurVS1 = STRINGIFY(
            uniform float pointRadius;  // point size in world space
            uniform float pointScale;   // scale to calculate size in pixels
            uniform float densityScale;
            uniform float densityOffset;
            void main()
            {
                vec4 wpos = vec4(gl_Vertex.xyz, 1.0);
                gl_Position = gl_ModelViewProjectionMatrix *wpos;

                vec4 eyeSpacePos = gl_ModelViewMatrix *wpos;
                float dist = length(eyeSpacePos.xyz);
                gl_PointSize = pointRadius * (pointScale / dist);

                gl_TexCoord[0] = gl_MultiTexCoord0;
                gl_TexCoord[1] = gl_Vertex.w;

                gl_FrontColor = gl_Color;
            }
    );


    const char* mblurGS =
            "#version 120\n"
            "#extension GL_EXT_geometry_shader4 : enable\n"
            STRINGIFY(
                    uniform float pointRadius;  // point size in world space
                    void main() {

                        // aging
                        float phase = gl_TexCoordIn[0][1].x;
                        float radius = pointRadius;

                        // eye space
                        vec3 pos = gl_PositionIn[0].xyz;
                        vec3 pos2 = gl_TexCoordIn[0][0].xyz;
                        vec3 motion = pos - pos2;
                        vec3 dir = normalize(motion);
                        float len = length(motion);

                        vec3 x = dir * radius;
                        vec3 view = normalize(-pos);
                        vec3 y = normalize(cross(dir, view)) * radius;
                        float facing = dot(view, dir);

                        // check for very small motion to avoid jitter
                        float threshold = 0.01;

                        if ((len < threshold) || (facing > 0.95) || (facing < -0.95)) {

                            pos2 = pos;

                            x = vec3(radius, 0.0, 0.0);

                            y = vec3(0.0, -radius, 0.0);

                        }

                        // output quad
                        gl_FrontColor = gl_FrontColorIn[0];
                        gl_TexCoord[0] = vec4(0, 0, 0, phase);
                        gl_TexCoord[1] = gl_PositionIn[0];
                        gl_Position = gl_ProjectionMatrix * vec4(pos + x + y, 1);
                        EmitVertex();

                        gl_TexCoord[0] = vec4(0, 1, 0, phase);
                        gl_TexCoord[1] = gl_PositionIn[0];
                        gl_Position = gl_ProjectionMatrix * vec4(pos + x - y, 1);
                        EmitVertex();

                        gl_TexCoord[0] = vec4(1, 0, 0, phase);
                        gl_TexCoord[1] = gl_PositionIn[0];
                        gl_Position = gl_ProjectionMatrix * vec4(pos2 - x + y, 1);
                        EmitVertex();

                        gl_TexCoord[0] = vec4(1, 1, 0, phase);
                        gl_TexCoord[1] = gl_PositionIn[0];
                        gl_Position = gl_ProjectionMatrix * vec4(pos2 - x - y, 1);
                        EmitVertex();
                    }
            );

    const char* particlePS = STRINGIFY(
            uniform float pointRadius;
            void main() {

                // calculate eye-space sphere normal from texture coordinates
                vec3 N;
                N.xy = gl_TexCoord[0].xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
                float r2 = dot(N.xy, N.xy);

                if (r2 > 1.0) discard;   // kill pixels outside circle
                N.z = sqrt(1.0 - r2);

                  float alpha = saturate(1.0 - r2);
                //float alpha = clamp((1.0 - r2), 0.0, 1.0);
                alpha *= gl_Color.w;

                gl_FragColor = vec4(gl_Color.xyz * alpha, alpha);
            }
    );
};
