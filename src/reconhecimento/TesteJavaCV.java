/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reconhecimento;

import org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;

/**
 *
 * @author luis
 */
public class TesteJavaCV
{
    public static void main(String args[])
    {
        FaceRecognizer r = EigenFaceRecognizer.create();
    }
}
