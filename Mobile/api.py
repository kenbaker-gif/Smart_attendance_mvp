import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:wakelock_plus/wakelock_plus.dart';
import 'verification_screen.dart';
import 'login_screen.dart';
import 'admin_screen.dart';
import 'stats_screen.dart';
import 'security_wrapper.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    WakelockPlus.enable();
  } catch (e) {
    debugPrint("Wakelock error: $e");
  }

  await dotenv.load(fileName: ".env");
  await Supabase.initialize(
    url: dotenv.env['SUPABASE_URL']!,
    anonKey: dotenv.env['SUPABASE_ANON_KEY']!,
  );

  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    debugPrint('Error in fetching the cameras: $e');
  }

  runApp(const AttendanceApp());
}

class AttendanceApp extends StatelessWidget {
  const AttendanceApp({super.key});

  @override
  Widget build(BuildContext context) {
    final session = Supabase.instance.client.auth.currentSession;
    final bool isLoggedIn = session != null;

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Smart Attendance',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: Colors.black,
        primaryColor: Colors.cyanAccent,
      ),
      builder: (context, child) {
        return ScaffoldMessenger(child: child!);
      },
      // ✅ Everyone lands on /home — admin buttons show up conditionally there
      initialRoute: isLoggedIn ? '/home' : '/login',
      routes: {
        '/login': (context) => LoginScreen(cameras: cameras),
        '/home': (context) => SecurityWrapper(
              child: VerificationScreen(cameras: cameras),
            ),
        '/admin': (context) => SecurityWrapper(
              isAdminRoute: true,
              child: AdminScreen(cameras: cameras),
            ),
        '/stats': (context) => SecurityWrapper(
              isAdminRoute: true,
              child: const StatsScreen(),
            ),
      },
    );
  }
}