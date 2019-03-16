/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reconhecimento;

import java.io.File;
import java.nio.IntBuffer;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FisherFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.LBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_GRAYSCALE;//talvez seja esse -- parece que é esse mesmo
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 *
 * @author luis
 */
public class TreinamentoYale
{

    public static void main(String[] args)
    {
        File diretorio = new File("src\\yalefaces\\treinamento");
        File[] arquivos = diretorio.listFiles();

        MatVector fotos = new opencv_core.MatVector(arquivos.length);

        Mat rotulos = new opencv_core.Mat(arquivos.length, 1, CV_32SC1);

        IntBuffer rotulosBuffer = rotulos.createBuffer();

        int contador = 0;
        for (File imagem : arquivos)
        {
            Mat foto = imread(imagem.getAbsolutePath(), IMREAD_GRAYSCALE);

            int classe = Integer.parseInt(imagem.getName().substring(7, 9));
            resize(foto, foto, new Size(160, 160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);//------Informa a qual classe cada imagem pertence

            contador++;
        }

        //FaceRecognizer eigenfaces = EigenFaceRecognizer.create(10, 0);//numero de componentes e threshold(limite de confiança - distancia de vizinhos)
        FaceRecognizer eigenfaces = EigenFaceRecognizer.create();
        FaceRecognizer fisherfaces = FisherFaceRecognizer.create();
        FaceRecognizer lbph = LBPHFaceRecognizer.create();

        //treina e salva cada um dos classificadores
        eigenfaces.train(fotos, rotulos);
        eigenfaces.save("src\\recursos\\classificadorEigenFacesYale.yml");
        lbph.train(fotos, rotulos);
        lbph.save("src\\recursos\\classificadorLBPHFacesYale.yml");
        fisherfaces.train(fotos, rotulos);//só funciona se tiver mais de 1 rotulo(pessoa);
        fisherfaces.save("src\\recursos\\classificadorFisherFacesYale.yml");
    }
}
